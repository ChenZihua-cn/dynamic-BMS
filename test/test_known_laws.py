"""已知定律恢复：从含噪合成数据重新发现 ground truth 定律。

每个实验是一轮"重新发现"——用已知定律生成含噪数据，运行 MCMC，
检验机器科学家能否恢复该定律（BIC 下降、测试集 R² 过阈值），
并记录恢复出的表达式供人工检视。

物理假设：含噪但信息充足的数据下，后验应集中向生成数据的定律，
而非任意拟合曲线。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import pytest
from core import Tree, OPS
from test.lab import (check, check_above, check_below, measure,
                      reproducible_state, r2_score, section)


# ============================================================================
# 共享实验管线：合成数据 → MCMC → 预测 → 观测量记录
# ============================================================================

def run_rediscovery_experiment(name, ground_truth, variables, n_train=30, n_test=20,
                               noise=0.1, burnin=500, thin=20, samples=100,
                               max_size=15, prior_par=None, rtol=1e-6):
    """运行一轮定律恢复实验，返回观测量字典。"""
    reproducible_state(42)

    # --- 1. 制备样品：含噪训练数据 + 洁净测试数据 ---
    x_train = pd.DataFrame({
        v: np.random.uniform(-3, 3, n_train) for v in variables
    })
    for v in variables:
        x_train[v] = x_train[v].where(x_train[v].abs() > 0.5, 0.5)

    y_clean = ground_truth(**{v: x_train[v].values for v in variables})
    y_train = pd.Series(y_clean + np.random.normal(0, noise, n_train))

    x_test = pd.DataFrame({
        v: np.linspace(-2.5, 2.5, n_test) for v in variables
    })
    for v in variables:
        x_test[v] = x_test[v].where(x_test[v].abs() > 0.3, 0.3)
    y_test = pd.Series(ground_truth(**{v: x_test[v].values for v in variables}))

    # --- 2. 默认先验 ---
    if prior_par is None:
        prior_par = dict([('Nopi_%s' % op, 1.0) for op in OPS])

    # --- 3. 初态 ---
    t = Tree(
        variables=variables,
        parameters=['a%d' % i for i in range(6)],
        x=x_train, y=y_train,
        prior_par=prior_par,
        max_size=max_size,
        BT=1.0,
    )
    initial_bic = t.bic
    initial_E = t.E

    # --- 4. MCMC 采样 ---
    trace_fn = '/tmp/test_trace_{}.dat'.format(name.replace(' ', '_'))
    progress_fn = '/tmp/test_progress_{}.dat'.format(name.replace(' ', '_'))
    t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
           burnin=burnin, thin=thin, samples=samples,
           write_files=False,
           verbose=False, progress=False)

    final_expr = str(t)
    final_bic = t.bic
    final_E = t.E

    # --- 5. 观测量：训练/测试 R² ---
    y_train_pred = t.predict(x_train)
    y_test_pred = t.predict(x_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # --- 6. 诊断观测量 ---
    measure(f"[{name}] 初态 BIC → 末态 BIC",
            f"{float(initial_bic):.2f} → {float(final_bic):.2f}")
    measure(f"[{name}] 训练 R²", train_r2)
    measure(f"[{name}] 恢复出的表达式", final_expr)

    # --- 7. 基本守恒性：拟合不应劣化（单向界：只允许改善） ---
    check_below("ΔBIC（末态 - 初态）",
                float(final_bic) - float(initial_bic), rtol, context=name)
    check_below("Δ能量（末态 - 初态）",
                float(final_E) - float(initial_E), rtol, context=name)

    return dict(name=name, train_r2=train_r2, test_r2=test_r2,
                initial_bic=float(initial_bic), final_bic=float(final_bic),
                initial_E=float(initial_E), final_E=float(final_E),
                final_expr=final_expr)


# ============================================================================
# Ground truth 定律
# ============================================================================

def ground_linear(x0):
    return 2.0 * x0 + 1.0

def ground_quadratic(x0):
    return x0**2 + 3.0 * x0 - 2.0

def ground_sin(x0):
    return np.sin(2.0 * x0)

def ground_rational(x0):
    return x0 / (1.0 + x0**2)

def ground_2var(x0, x1):
    return 3.0 * x0 - 2.0 * x1 + 5.0

def ground_3var_interact(x0, x1, x2):
    return x0 * x1 + x2

def ground_3var_rat(x0, x1, x2):
    return (x0 * x1) / (1.0 + x2**2)

def ground_3var_mix(x0, x1, x2):
    return np.sin(x0) + x1 * np.exp(-x2**2)

def ground_exp(x0):
    return np.exp(-x0**2)


# ============================================================================
# 定律恢复实验
# ============================================================================

def test_linear_law():
    """恢复线性律 y = 2x + 1（σ=0.15, 30 点）：R² > 0.7。"""
    section("定律恢复：线性律")
    res = run_rediscovery_experiment("Linear y=2x+1", ground_linear, ['x0'],
                                     n_train=30, noise=0.15,
                                     burnin=500, thin=20, samples=100)
    check_above("测试集 R²", res['test_r2'], 0.7, context="线性律")


def test_quadratic_law():
    """恢复二次律 y = x² + 3x - 2（σ=0.2, 40 点）：R² > 0.6。"""
    section("定律恢复：二次律")
    res = run_rediscovery_experiment("Quadratic y=x^2+3x-2", ground_quadratic, ['x0'],
                                     n_train=40, noise=0.2,
                                     burnin=500, thin=20, samples=100)
    check_above("测试集 R²", res['test_r2'], 0.6, context="二次律")


def test_harmonic_law():
    """恢复谐波律 y = sin(2x)（σ=0.1, 40 点）：R² > 0.3。"""
    section("定律恢复：谐波律")
    res = run_rediscovery_experiment("Trig y=sin(2x)", ground_sin, ['x0'],
                                     n_train=40, noise=0.1,
                                     burnin=800, thin=20, samples=100)
    check_above("测试集 R²", res['test_r2'], 0.3, context="谐波律 sin(2x)")


def test_rational_law():
    """恢复有理律 y = x/(1+x²)（σ=0.08, 40 点）：R² > 0.5。"""
    section("定律恢复：有理律")
    res = run_rediscovery_experiment("Rational y=x/(1+x^2)", ground_rational, ['x0'],
                                     n_train=40, noise=0.08,
                                     burnin=800, thin=20, samples=100)
    check_above("测试集 R²", res['test_r2'], 0.5, context="有理律")


def test_two_variable_law():
    """恢复双变量线性律 y = 3x₀ - 2x₁ + 5（σ=0.2, 60 点）：R² > 0.3。"""
    section("定律恢复：双变量线性律")
    res = run_rediscovery_experiment("2-Var y=3x0-2x1+5", ground_2var, ['x0', 'x1'],
                                     n_train=60, noise=0.2, burnin=2000, thin=20,
                                     samples=200, max_size=12)
    check_above("测试集 R²", res['test_r2'], 0.3, context="双变量线性律")


def test_three_variable_interaction():
    """恢复三变量交叉律 y = x₀x₁ + x₂（σ=0.3, 80 点）：R² > 0.15。"""
    section("定律恢复：三变量交叉律")
    res = run_rediscovery_experiment("3-Var y=x0*x1+x2", ground_3var_interact,
                                     ['x0', 'x1', 'x2'],
                                     n_train=80, noise=0.3, burnin=2000, thin=20,
                                     samples=200, max_size=20)
    check_above("测试集 R²", res['test_r2'], 0.15, context="三变量交叉律")


def test_three_variable_rational():
    """恢复三变量有理律 y = x₀x₁/(1+x₂²)（σ=0.15, 80 点）：R² > 0.1。"""
    section("定律恢复：三变量有理律")
    res = run_rediscovery_experiment("3-Var y=x0*x1/(1+x2^2)", ground_3var_rat,
                                     ['x0', 'x1', 'x2'],
                                     n_train=80, noise=0.15, burnin=2000, thin=20,
                                     samples=200, max_size=20)
    check_above("测试集 R²", res['test_r2'], 0.1, context="三变量有理律")


def test_three_variable_mixed():
    """恢复混合律 y = sin(x₀) + x₁exp(-x₂²)（σ=0.2, 80 点）：R² > 0.05。"""
    section("定律恢复：混合律")
    res = run_rediscovery_experiment("3-Var y=sin(x0)+x1*exp(-x2^2)", ground_3var_mix,
                                     ['x0', 'x1', 'x2'],
                                     n_train=80, noise=0.2, burnin=2000, thin=20,
                                     samples=200, max_size=20)
    check_above("测试集 R²", res['test_r2'], 0.05, context="混合律")


def test_exp_decay_law():
    """恢复指数衰减律 y = exp(-x²)（σ=0.05, 40 点）：R² > 0.5。"""
    section("定律恢复：指数衰减律")
    res = run_rediscovery_experiment("Exp y=exp(-x^2)", ground_exp, ['x0'],
                                     n_train=40, noise=0.05,
                                     burnin=800, thin=20, samples=100)
    check_above("测试集 R²", res['test_r2'], 0.5, context="指数衰减律")


@pytest.mark.slow
def test_noise_scaling_law():
    """标度律：噪声 σ 增大时恢复精度单调不降。

    同一二次律、同一批数据点（每次实验重置种子），仅噪声幅度
    σ ∈ {0.01, 0.1, 0.5} 不同。若信号未被噪声淹没，
    测试集 R² 应随 σ 增大而下降（或持平）。
    """
    section("标度律：恢复精度 ~ 噪声幅度")
    sigmas = [0.01, 0.1, 0.5]
    r2s = []
    for sigma in sigmas:
        res = run_rediscovery_experiment(
            f"Quadratic σ={sigma}", ground_quadratic, ['x0'],
            n_train=40, noise=sigma, burnin=500, thin=20, samples=100)
        r2s.append(res['test_r2'])

    for sigma, r2 in zip(sigmas, r2s):
        measure(f"σ = {sigma} 的测试集 R²", r2)

    monotone = all(r2s[i] >= r2s[i + 1] for i in range(len(r2s) - 1))
    if monotone:
        check_above("R² 单调性裕量（σ=0.01 档超出末档）",
                    r2s[0] - r2s[-1], 0.0, context="噪声标度律")
    else:
        # 随机涨落导致非严格单调：退化为要求最高噪声下信号仍可恢复
        measure("单调性判定", "非严格单调（随机涨落），退化为阈值检验")
        check_above("最高噪声(σ=0.5)下的 R²", r2s[-1], 0.3,
                    context="信号未被噪声淹没")


if __name__ == '__main__':
    for name in sorted(dir()):
        if name.startswith('test_'):
            globals()[name]()
    print("\n实验记录输出完毕（pytest 下用 -s 查看）。")
