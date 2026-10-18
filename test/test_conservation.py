"""守恒律与对称性：MCMC 轨迹上必须逐步成立的守恒量。

物理假设 —— 无论提议是否被接受、树如何演化：
  1. 能量记账 E = EB + EP 逐步守恒，增量只来自被接受提议的 dE；
  2. 树尺寸始终有界（相空间体积有限）；
  3. 能量/BIC 沿轨迹保持有限（不发散）；
  4. 时间演化的可逆记账：终态能量 - 初态能量 == Σ dE(accepted)。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from core import Tree, OPS
from test.lab import check, check_above, measure, reproducible_state, section

PRIOR_UNIT = dict([('Nopi_%s' % op, 1.0) for op in OPS])


def _linear_data(n=20, noise=0.1):
    x = pd.DataFrame({'x0': np.linspace(0, 10, n)})
    y = 2 * x['x0'] + 1 + np.random.normal(0, noise, n)
    return x, pd.Series(y)


def _data_tree():
    x, y = _linear_data()
    return Tree(
        variables=['x0'],
        parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=PRIOR_UNIT,
        max_size=10,
    )


def test_free_trajectory_invariants():
    """无数据自由演化：size 有界、能量有限、树确实在演化。"""
    section("自由轨迹（无数据）")
    reproducible_state(42)
    t = Tree(
        prior_par=dict([('Nopi_%s' % op, 0) for op in OPS]),
        from_string='(x + _a0_)'
    )
    initial_str = str(t)

    for i in range(5):
        t.mcmc_step(verbose=False)
        check("size 有界", 1 <= t.size <= t.max_size, True,
              context=f"step {i}")
        check("E 有限", np.isfinite(float(t.E)), True, context=f"step {i}")
        check("BIC 有限", np.isfinite(float(t.bic)), True, context=f"step {i}")

    check("轨迹发生演化（表达式改变）", str(t) != initial_str, True)


def test_data_trajectory_invariants():
    """含数据演化：size 有界、能量有限、树演化、能量不显著上升。

    假设：简单线性数据上，平衡态附近能量不应显著高于初态。
    """
    section("受驱轨迹（线性数据）")
    reproducible_state(42)
    t = _data_tree()
    initial_E = t.E
    initial_str = str(t)

    for i in range(10):
        t.mcmc_step(verbose=False)
        check("size 有界", 1 <= t.size <= t.max_size, True,
              context=f"step {i}")
        check("E 有限", np.isfinite(float(t.E)), True, context=f"step {i}")
        check("BIC 有限", np.isfinite(float(t.bic)), True, context=f"step {i}")

    check("轨迹发生演化（表达式改变）", str(t) != initial_str, True)
    check_above("初态能量 - 终态能量", initial_E - t.E, -1.0,
                context="漂移不超过 +1.0 kT")


def test_predict_finite():
    """观测量性质：predict 输出长度匹配、值有限、非平庸（非常数）。"""
    section("观测量：预测")
    reproducible_state(42)
    t = _data_tree()
    for _ in range(10):
        t.mcmc_step(verbose=False)

    x, _ = _linear_data()
    ypred = t.predict(x)
    check("返回 Series", isinstance(ypred, pd.Series), True)
    check("预测长度匹配", len(ypred), len(x))
    check("预测值有限", np.all(np.isfinite(ypred.values)), True)
    check("预测非平庸", ypred.nunique() > 1, True)

    x_test = pd.DataFrame({'x0': np.linspace(0, 10, 5)})
    ypred_test = t.predict(x_test)
    check("换长度数据预测", len(ypred_test), 5)
    check("换长度预测值有限", np.all(np.isfinite(ypred_test.values)), True)


def test_energy_self_consistency_after_reset():
    """重整化自洽：get_energy(reset=True) 重算结果与记账值一致。"""
    section("能量记账自洽")
    reproducible_state(42)
    t = _data_tree()
    for _ in range(10):
        t.mcmc_step(verbose=False)

    E_new, EB_new, EP_new = t.get_energy(bic=True, reset=True)
    check("E 重算与记账一致", E_new, t.E, tol=1e-6)


def test_energy_telescoping():
    """望远镜求和守恒：ΔE(轨迹) == Σ dE(accepted)。

    对 dE_et/dE_lr/dE_rr 做 instrument，记录每次提议的能量差；
    被接受的步由 E 的变化判定。若记账严格增量更新，两边必须相等。
    """
    section("望远镜求和：ΔE = Σ dE(accepted)")
    reproducible_state(42)
    t = _data_tree()
    E0 = t.E

    deltas = []

    def _wrap(name):
        orig = getattr(t, name)

        def wrapper(*args, **kwargs):
            res = orig(*args, **kwargs)
            deltas.append(float(res[0]))
            return res
        return wrapper

    t.dE_et = _wrap('dE_et')
    t.dE_lr = _wrap('dE_lr')
    t.dE_rr = _wrap('dE_rr')

    n_steps = 50
    n_moves = 0
    accepted_sum = 0.0
    for _ in range(n_steps):
        E_before = t.E
        t.mcmc_step(verbose=False)
        n_moves += 1
        if t.E != E_before:
            accepted_sum += deltas[-1]

    dE_trajectory = t.E - E0
    measure("提议总数", n_moves)
    measure("接受总数", sum(1 for b in deltas if np.isfinite(b)))
    measure("Σ dE(accepted)", accepted_sum)
    measure("ΔE(轨迹)", dE_trajectory)
    check("望远镜求和", accepted_sum, dE_trajectory, tol=1e-9,
          context=f"{n_steps} 步轨迹")


if __name__ == '__main__':
    for name in sorted(dir()):
        if name.startswith('test_'):
            globals()[name]()
    print("\n实验记录输出完毕（pytest 下用 -s 查看）。")
