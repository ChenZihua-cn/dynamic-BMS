"""实验记录框架 —— 把测试写成物理实验。

每个实验函数用 check() 记录测量量并与理论预期比较，用 measure()
记录诊断观测量（如恢复出的表达式、R²）。超差时 check() 抛出带物理
上下文的 AssertionError；pytest 照常收集、照常报告。

    from test.lab import check, measure, section

    def test_energy_bookkeeping():
        \"\"\"假设：MCMC 轨迹上 E = EB + EP 逐步守恒。\"\"\"
        section("能量记账守恒")
        ...
        check("E = EB + EP", dev, 0.0, tol=1e-9, context=f"step {i}")

也可以直接 `python test/test_xxx.py` 运行单个实验文件查看实验记录
（文件需自带 __main__ 块，见 test_calibration.py 范本）。
"""
import random

import numpy as np

__all__ = ["reproducible_state", "section", "measure", "check",
           "check_above", "check_below", "r2_score"]


def reproducible_state(seed=42):
    """制备可重复的实验初态：同时固定 numpy 与 stdlib 两个随机源。

    core 的 MCMC 掷骰子用 stdlib random（core/mcmc.py），而测试数据
    生成用 np.random——只固定其中之一，轨迹不可复现。
    """
    np.random.seed(seed)
    random.seed(seed)


def _fmt(x):
    try:
        return f"{float(x):.6g}"
    except (TypeError, ValueError):
        return str(x)


def _is_number(x):
    """任何可安全转为 float 的量都按数值处理（含 sympy Float / numpy 标量）。"""
    if isinstance(x, bool):
        return False
    try:
        float(x)
    except (TypeError, ValueError):
        return False
    return True


def section(title):
    """打印实验分区标题。"""
    print(f"\n─── {title} ───")


def measure(quantity, value, unit=""):
    """记录一个观测量（不断言），返回 value。"""
    print(f"  [观测] {quantity} = {_fmt(value)}{unit}")
    return value


def check(quantity, measured, expected, *, tol=None, rtol=None,
          unit="", context=""):
    """实验检验：measured 与 expected 的偏差必须在容差内。

    tol: 绝对容差；rtol: 相对容差（容差 = rtol * |expected|）。
    数值型至少给其一（默认精确相等）；非数值型做相等性检验。
    超差抛 AssertionError。
    """
    where = f"（{context}）" if context else ""
    if _is_number(measured) and _is_number(expected):
        measured, expected = float(measured), float(expected)
        if tol is None and rtol is None:
            tol = 0.0
        if tol is None:
            tol = rtol * abs(expected)
        dev = abs(float(measured) - float(expected))
        ok = dev <= tol
        status = "在容差内" if ok else "✗ 超差"
        print(f"  [检验] {quantity}{where}: 期望 {_fmt(expected)}, "
              f"实测 {_fmt(measured)}{unit}, 偏差 {_fmt(dev)}, "
              f"容差 {_fmt(tol)} — {status}")
        if not ok:
            raise AssertionError(
                f"{quantity}{where}: 期望 {_fmt(expected)}, "
                f"实测 {_fmt(measured)}, 偏差 {_fmt(dev)} 超出容差 {_fmt(tol)}")
    else:
        ok = measured == expected
        print(f"  [检验] {quantity}{where}: 期望 {expected!r}, "
              f"实测 {measured!r} — {'在容差内' if ok else '✗ 不符'}")
        if not ok:
            raise AssertionError(
                f"{quantity}{where}: 期望 {expected!r}, 实测 {measured!r}")
    return measured


def check_above(quantity, measured, threshold, *, unit="", context=""):
    """实验检验：measured 必须大于 threshold（如 R² 恢复阈值）。"""
    where = f"（{context}）" if context else ""
    ok = measured > threshold
    print(f"  [检验] {quantity}{where}: 实测 {_fmt(measured)}{unit}, "
          f"须 > {_fmt(threshold)} — {'在阈上' if ok else '✗ 低于阈值'}")
    if not ok:
        raise AssertionError(
            f"{quantity}{where}: 实测 {_fmt(measured)} 未超过阈值 {_fmt(threshold)}")
    return measured


def check_below(quantity, measured, threshold, *, unit="", context=""):
    """实验检验：measured 必须小于 threshold（如拒绝率上限）。"""
    where = f"（{context}）" if context else ""
    ok = measured < threshold
    print(f"  [检验] {quantity}{where}: 实测 {_fmt(measured)}{unit}, "
          f"须 < {_fmt(threshold)} — {'在阈下' if ok else '✗ 超出阈值'}")
    if not ok:
        raise AssertionError(
            f"{quantity}{where}: 实测 {_fmt(measured)} 超出阈值 {_fmt(threshold)}")
    return measured


def r2_score(y_true, y_pred):
    """决定系数 R² —— 拟合优度的标准观测量。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
