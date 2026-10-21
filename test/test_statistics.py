"""统计物理观测：把 MCMC 采样链当作统计物理系统来测量。

观测量：
  1. 宪法门拒绝率 —— 硬约束是否阻断相空间探索（含 plan_origin.md
     的相变决策规则：<50% 维持硬拒绝，50-80% 保留并记录，>80% 升级
     为量纲感知提议）；
  2. 细致平衡的方向性 —— 能量下降提议的接受率应显著高于上升提议；
  3. 轨迹记录完整性 —— 采样链的实验记录可写、可读、结构正确；
  4. 系综预测与关联 —— 预测-实测关联、链的涨落（探索未冻结）。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core import Tree, Parallel, OPS
from experts.constitution.dimensional import DimensionalConstitution
from experts.parliament.occam import OccamPrior
from test.lab import (check, check_above, check_below, measure,
                      reproducible_state, section)

DIM_L = (1, 0, 0, 0, 0, 0, 0)
DIM_G = (1, 0, -2, 0, 0, 0, 0)
DIM_D = (0, 0, 0, 0, 0, 0, 0)
PENDULUM_DIMS = {'L': DIM_L, 'g': DIM_G, 'm': (0, 1, 0, 0, 0, 0, 0),
                 'theta0': DIM_D}

PRIOR_UNIT = dict([('Nopi_%s' % op, 1.0) for op in OPS])


# ---------------------------------------------------------------------------
# 仪器：提议跟踪（在 dE_* 上装探测器）
# ---------------------------------------------------------------------------

def _instrument_proposals(tree):
    """Wrap tree's dE_* and check_constitution to capture proposal outcomes.

    Returns *(proposal_log, cleanup_fn)*. Each entry is
    ``{'method': str, 'dE': float, 'constitution_failed': bool}``.
    """
    proposal_log = []
    orig_dE_et = tree.dE_et
    orig_dE_lr = tree.dE_lr
    orig_dE_rr = tree.dE_rr
    orig_check = tree.check_constitution

    def _make_wrapper(orig_fn, name):
        def wrapper(*args, **kwargs):
            check_calls = []

            def tracking_check():
                res = orig_check()
                check_calls.append(res)
                return res

            tree.check_constitution = tracking_check
            try:
                result = orig_fn(*args, **kwargs)
            finally:
                tree.check_constitution = orig_check

            dE_val = result[0] if isinstance(result, tuple) else result
            proposal_log.append({
                'method': name,
                'dE': float(dE_val),
                'constitution_failed': any(not c for c in check_calls),
            })
            return result
        return wrapper

    tree.dE_et = _make_wrapper(orig_dE_et, 'dE_et')
    tree.dE_lr = _make_wrapper(orig_dE_lr, 'dE_lr')
    tree.dE_rr = _make_wrapper(orig_dE_rr, 'dE_rr')

    def cleanup():
        tree.dE_et = orig_dE_et
        tree.dE_lr = orig_dE_lr
        tree.dE_rr = orig_dE_rr

    return proposal_log, cleanup


def _run_and_classify(tree, proposal_log, n_steps):
    """Run *n_steps* of MCMC and classify each proposal outcome.

    Returns dict with keys: total, accepted, constitution_reject,
    canonical_reject.
    """
    stats = {'total': 0, 'accepted': 0,
             'constitution_reject': 0, 'canonical_reject': 0}

    for _ in range(n_steps):
        E_before = tree.E
        tree.mcmc_step(verbose=False)
        E_after = tree.E

        entry = proposal_log.pop(0)
        stats['total'] += 1
        if E_after != E_before:
            stats['accepted'] += 1
        elif entry['constitution_failed']:
            stats['constitution_reject'] += 1
        elif np.isinf(entry['dE']):
            stats['canonical_reject'] += 1

    return stats


def _print_rejection_report(stats, n_steps):
    """Pretty-print the rejection rate statistics."""
    total = stats['total']
    fmt = lambda n: f"{n:>6d}  ({n / total:.1%})"
    const_rate = stats['constitution_reject'] / total

    print(f"\n{'=' * 60}")
    print(f"Constitution Gate Rejection Rate Report ({n_steps} steps)")
    print(f"{'=' * 60}")
    print(f"  Total proposals:       {stats['total']:>6d}")
    print(f"  Accepted:              {fmt(stats['accepted'])}")
    print(f"  Constitution-rejected: {fmt(stats['constitution_reject'])}")
    print(f"  Canonical-rejected:    {fmt(stats['canonical_reject'])}")
    other = total - stats['accepted'] - stats['constitution_reject'] - stats['canonical_reject']
    print(f"  Other (MCMC prob):     {other:>6d}")
    print(f"{'=' * 60}")
    if const_rate < 0.50:
        decision = "keep 5-A (hard rejection)"
    elif const_rate < 0.80:
        decision = "keep 5-A, record data"
    else:
        decision = "UPGRADE to 5-D (dimension-aware proposal)"
    print(f"  Decision: {decision}")
    print(f"{'=' * 60}\n")


def _pendulum_tree(with_parliament=False):
    reproducible_state(42)
    n = 30
    L_vals = np.linspace(0.5, 2.0, n)
    g_val = 9.81
    T_true = 2 * np.pi * np.sqrt(L_vals / g_val)
    T_obs = T_true + np.random.normal(0, 0.02, n)
    x = pd.DataFrame({'L': L_vals, 'g': [g_val] * n})
    y = pd.Series(T_obs)

    kwargs = {}
    if with_parliament:
        kwargs['parliaments'] = [OccamPrior()]
    tree = Tree(
        variables=['L', 'g'], parameters=['a', 'b'],
        dimensions=PENDULUM_DIMS,
        constitutions=[DimensionalConstitution()],
        x=x, y=y, max_size=30,
        from_string='(_a0_ * sqrt((L / g)))',
        **kwargs
    )
    return tree


# ============================================================================
# 实验 1：宪法门拒绝率（相变决策规则）
# ============================================================================

def test_constitution_rejection_rate():
    """测量宪法硬拒绝率：须 < 80%，否则硬约束阻断探索（升级 5-D）。"""
    section("宪法门拒绝率观测")
    tree = _pendulum_tree()
    check("初态通过宪法检查", tree.check_constitution(), True)

    proposal_log, cleanup = _instrument_proposals(tree)
    n_steps = 200
    stats = _run_and_classify(tree, proposal_log, n_steps)
    cleanup()

    _print_rejection_report(stats, n_steps)

    const_rate = stats['constitution_reject'] / stats['total']
    check_below("宪法门拒绝率", const_rate, 0.80,
                context=f"{n_steps} 步轨迹")


def test_constitution_rejection_rate_with_parliament():
    """议会不参与拒绝统计：宪法先于议会执行，拒绝率应不受影响。"""
    section("宪法门拒绝率观测（含议会）")
    tree = _pendulum_tree(with_parliament=True)
    check("初态通过宪法检查", tree.check_constitution(), True)

    proposal_log, cleanup = _instrument_proposals(tree)
    stats = _run_and_classify(tree, proposal_log, n_steps=100)
    cleanup()

    _print_rejection_report(stats, 100)

    const_rate = stats['constitution_reject'] / stats['total']
    check_below("宪法门拒绝率", const_rate, 0.80,
                context="含 OccamPrior")


# ============================================================================
# 实验 2：细致平衡的方向性
# ============================================================================

def test_metropolis_directionality():
    """细致平衡方向性：r_accept(ΔE<0) 应显著高于 r_accept(ΔE>0)。

    提议比修正项（nif·ωi·sf / nfi·ωf·si）使精确频率匹配不成立，
    故只检验方向性，不检验精确 Metropolis 比率。
    """
    section("细致平衡方向性")
    reproducible_state(42)
    x = pd.DataFrame({'x0': np.linspace(0, 10, 20)})
    y = pd.Series(2 * x['x0'] + 1 + np.random.normal(0, 0.1, 20))
    tree = Tree(
        variables=['x0'], parameters=['a0', 'a1'],
        x=x, y=y, prior_par=PRIOR_UNIT, max_size=10,
    )

    log = []

    def _wrap(name):
        orig = getattr(tree, name)

        def wrapper(*args, **kwargs):
            res = orig(*args, **kwargs)
            log.append(float(res[0]))
            return res
        return wrapper

    tree.dE_et = _wrap('dE_et')
    tree.dE_lr = _wrap('dE_lr')
    tree.dE_rr = _wrap('dE_rr')

    n_steps = 300
    outcomes = []
    for _ in range(n_steps):
        E_before = tree.E
        tree.mcmc_step(verbose=False)
        outcomes.append(tree.E != E_before)

    dE = np.array(log)
    acc = np.array(outcomes)
    valid = np.isfinite(dE) & (dE != 0.0)
    downhill = valid & (dE < 0)
    uphill = valid & (dE > 0)

    n_down, n_up = int(downhill.sum()), int(uphill.sum())
    r_down = float(acc[downhill].mean()) if n_down else float('nan')
    r_up = float(acc[uphill].mean()) if n_up else float('nan')
    measure(" downhill 提议数 / 接受率", f"{n_down} / {r_down:.3f}")
    measure("uphill 提议数 / 接受率", f"{n_up} / {r_up:.3f}")

    if n_down >= 10 and n_up >= 10:
        check_above("downhill 接受率高出 uphill", r_down - r_up, 0.0,
                    context=f"{n_steps} 步轨迹")
    else:
        measure("方向性检验", "样本不足（某桶 < 10），仅记录不上检验")


# ============================================================================
# 实验 3：轨迹记录完整性
# ============================================================================

def test_trace_file_write():
    """采样链的实验记录：trace/progress 文件行数与 JSON 结构正确。"""
    section("轨迹记录完整性")
    reproducible_state(99)
    x = pd.DataFrame({'x0': np.linspace(0, 5, 15)})
    y = 1.5 * x['x0'] + np.random.normal(0, 0.2, 15)

    t = Tree(
        variables=['x0'], parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=PRIOR_UNIT,
        max_size=10,
    )
    trace_fn = '/tmp/test_trace_output.dat'
    progress_fn = '/tmp/test_progress_output.dat'
    t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
           burnin=200, thin=10, samples=30,
           write_files=True, reset_files=True,
           verbose=False, progress=False)

    with open(trace_fn, 'r') as f:
        trace_lines = f.readlines()
    check("trace 行数", len(trace_lines), 30)

    for i, line in enumerate(trace_lines):
        record = json.loads(line)
        check("记录字段数", len(record), 6, context=f"line {i}")
        check("sample id 为整数", isinstance(record[0], int), True,
              context=f"line {i}")
        check("BIC 为浮点", isinstance(record[1], float), True,
              context=f"line {i}")
        check("Energy 为浮点", isinstance(record[2], float), True,
              context=f"line {i}")

    with open(progress_fn, 'r') as f:
        progress_lines = f.readlines()
    check("progress 行数", len(progress_lines), 30)


def test_trace_predict():
    """系综预测：trace_predict 对每个采样点输出预测，全部有限。"""
    section("系综预测")
    reproducible_state(123)
    x = pd.DataFrame({'x0': np.linspace(0, 5, 20)})
    y = 2.0 * x['x0'] + 1.0 + np.random.normal(0, 0.3, 20)

    t = Tree(
        variables=['x0'], parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=PRIOR_UNIT,
        max_size=10,
        from_string='(x0 + _a0_)',
    )
    trace_fn = '/tmp/test_trace_predict.dat'
    progress_fn = '/tmp/test_progress_predict.dat'
    t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
           burnin=100, thin=10, samples=10,
           write_files=True, reset_files=True,
           verbose=False, progress=False)

    x_test = pd.DataFrame({'x0': np.linspace(0, 5, 10)})
    ypred_trace = t.trace_predict(x_test, samples=10, burnin=0)
    check("返回 DataFrame", isinstance(ypred_trace, pd.DataFrame), True)
    check("预测形状 (样本, 采样点)", ypred_trace.shape, (10, 10))
    check("预测值有限", np.all(np.isfinite(ypred_trace.values)), True)


# ============================================================================
# 实验 4：链的涨落与预测-实测关联
# ============================================================================

def test_chain_fluctuation():
    """链在探索：BIC/Energy 轨迹有限且非常数（未冻结在局部极小）。"""
    section("链的涨落：BIC/Energy 轨迹")
    reproducible_state(7)
    x = pd.DataFrame({'x0': np.linspace(-2, 2, 40)})
    y = pd.Series(2.0 * x['x0'] + 1.0 + np.random.normal(0, 0.2, 40))

    t = Tree(
        variables=['x0'], parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=PRIOR_UNIT,
        max_size=10,
        from_string='((x0 * _a0_) + _a1_)',
    )
    trace_fn = '/tmp/test_viz_trace.dat'
    progress_fn = '/tmp/test_viz_progress.dat'
    t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
           burnin=200, thin=10, samples=50,
           write_files=True, reset_files=True,
           verbose=False, progress=False)

    bic_trace, E_trace = [], []
    with open(trace_fn, 'r') as f:
        for line in f:
            rec = json.loads(line)
            bic_trace.append(rec[1])
            E_trace.append(rec[2])

    bic_trace = np.array(bic_trace)
    E_trace = np.array(E_trace)
    measure("BIC 轨迹均值 ± 标准差", f"{bic_trace.mean():.2f} ± {bic_trace.std():.2f}")
    measure("Energy 轨迹均值 ± 标准差", f"{E_trace.mean():.2f} ± {E_trace.std():.2f}")

    check("轨迹样本数", len(bic_trace), 50)
    check("BIC 轨迹有限", np.all(np.isfinite(bic_trace)), True)
    check("Energy 轨迹有限", np.all(np.isfinite(E_trace)), True)
    check_above("BIC 轨迹涨落（非冻结）", np.std(bic_trace), 0.0)
    check_above("Energy 轨迹涨落（非冻结）", np.std(E_trace), 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(bic_trace)
    ax1.set_xlabel('MCMC sample')
    ax1.set_ylabel('BIC')
    ax1.set_title('BIC trace ({})'.format(t))
    ax2.plot(E_trace)
    ax2.set_xlabel('MCMC sample')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy trace ({})'.format(t))
    plt.tight_layout()
    fig.savefig('/tmp/test_viz_trace.png', dpi=100)
    plt.close(fig)


def test_prediction_correlation():
    """预测-实测关联：预测与真值的相关系数须 > 0.5。"""
    section("预测-实测关联")
    reproducible_state(7)
    x = pd.DataFrame({'x0': np.linspace(-2, 2, 40)})
    y = pd.Series(2.0 * x['x0'] + 1.0 + np.random.normal(0, 0.2, 40))

    t = Tree(
        variables=['x0'], parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=PRIOR_UNIT,
        max_size=10,
        from_string='((x0 * _a0_) + _a1_)',
    )
    trace_fn = '/tmp/test_viz_trace.dat'
    progress_fn = '/tmp/test_viz_progress.dat'
    t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
           burnin=200, thin=10, samples=50,
           write_files=True, reset_files=True,
           verbose=False, progress=False)

    ypred = t.predict(x)

    check("返回 Series", isinstance(ypred, pd.Series), True)
    check("预测长度匹配", len(ypred), len(x))
    check("预测值有限", np.all(np.isfinite(ypred.values)), True)

    corr = np.corrcoef(ypred.values, y.values)[0, 1]
    measure("预测-实测相关系数", corr)
    check_above("预测-实测相关", corr, 0.5)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ypred, y, alpha=0.7)
    lo = min(y.min(), ypred.min())
    hi = max(y.max(), ypred.max())
    pad = 0.1 * (hi - lo) if hi > lo else 0.5
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', linewidth=1,
            label='y = x')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Predictions vs Actual ({})'.format(t))
    ax.legend()
    fig.savefig('/tmp/test_viz_pred_vs_actual.png', dpi=100)
    plt.close(fig)


# ============================================================================
# 实验 8-11：并行回火阶梯（replica exchange）
# ============================================================================

LADDER_TS = [1] + [1.04 ** k for k in range(1, 8)]


def _ladder(Ts=LADDER_TS, seed=7):
    reproducible_state(seed)
    n = 30
    L_vals = np.linspace(0.5, 2.0, n)
    x = pd.DataFrame({'L': L_vals, 'g': [9.81] * n})
    y = pd.Series(2 * np.pi * np.sqrt(L_vals / 9.81)
                  + np.random.normal(0, 0.02, n))
    return Parallel(
        list(Ts), variables=['L', 'g'], parameters=['a', 'b'],
        dimensions=PENDULUM_DIMS,
        constitutions=[DimensionalConstitution()],
        x=x, y=y, max_size=30, prior_par=PRIOR_UNIT,
    )


def test_ladder_requires_a_real_ladder():
    """构造前提：少于 2 个温度或缺 BT=1 冷副本都必须点名拒绝，而不是跑起来才炸。"""
    section("回火阶梯的构造前提")
    for Ts, needle in ([[1], 'at least 2 temperatures'],
                       [[2, 5], 'must include BT = 1.0']):
        msg = ''
        try:
            _ladder(Ts=Ts)
        except ValueError as e:
            msg = str(e)
        check("拒绝 %s" % (Ts,), needle in msg, True)


def test_ladder_exchange_consistency():
    """交换只搬表达式、不搬温度：槽位 BT、能量记账、退火归位三处必须自洽。"""
    section("并行回火阶梯：交换自洽性")
    p = _ladder()
    check("阶梯副本数", len(p.trees), len(p.Ts))
    check("拟合缓存跨副本共享",
          all(t.fit_par is p.t1.fit_par for t in p.trees.values()), True)
    check("简并代表表跨副本共享",
          all(t.representative is p.t1.representative
              for t in p.trees.values()), True)

    n_steps = 400
    n_acc, n_bad_slot, n_bad_energy = 0, 0, 0
    cold_forms = {str(p.t1)}
    for i in range(n_steps):
        p.mcmc_step()
        if p.tree_swap() != (None, None):
            n_acc += 1
        for BT, t in p.trees.items():
            if abs(t.BT - BT) > 1e-12:
                n_bad_slot += 1
            if abs(float(t.E) - float(t.get_energy(reset=False)[0])) \
                    > 1e-6 * max(1.0, abs(float(t.E))):
                n_bad_energy += 1
        cold_forms.add(str(p.t1))

    check_above("阶梯交换接受率", n_acc / float(n_steps), 0.1,
                context='%d 次尝试' % n_steps)
    check("温度槽位自洽违例数", n_bad_slot, 0)
    check("能量记账自洽违例数", n_bad_energy, 0)
    check_above("冷副本访问过的不同表达式数", len(cold_forms), 1,
                context='交换须能传到 BT=1')

    p.anneal(n=20, factor=3)
    check("退火后阶梯归位违例数",
          sum(0 if abs(t.BT - BT) < 1e-12 else 1
              for BT, t in p.trees.items()), 0)


def test_ladder_trace_matches_single_temperature_schema():
    """回火迹线必须能被读单温度迹线的分析代码原样消费（6 字段、行数 = samples）。"""
    section("回火迹线与单温度格式兼容")
    p = _ladder(Ts=[1, 1.5, 2.5, 4.0])
    trace_fn = '/tmp/test_ladder_trace.dat'
    progress_fn = '/tmp/test_ladder_progress.dat'
    n_annealed = p.mcmc(tracefn=trace_fn, progressfn=progress_fn,
                        burnin=300, thin=5, samples=25,
                        progress=False, anneal_after=0)

    with open(trace_fn) as f:
        lines = f.readlines()
    check("回火 trace 行数", len(lines), 25)
    for i, line in enumerate(lines):
        record = json.loads(line)
        check("记录字段数", len(record), 6, context=f"line {i}")
        check("sample id 为整数", isinstance(record[0], int), True,
              context=f"line {i}")
        check("BIC 为浮点", isinstance(record[1], float), True,
              context=f"line {i}")
        check("Energy 为浮点", isinstance(record[2], float), True,
              context=f"line {i}")
    with open(progress_fn) as f:
        check("回火 progress 行数", len(f.readlines()), 25)
    check("退火触发次数", n_annealed, 0)


def test_ladder_anneal_on_stall():
    """停滞判据按 MCMC 步数算：用一副温度跨度过大、几乎换不动的阶梯验证退火真的触发。"""
    section("回火阶梯：停滞退火")
    # BT=1 与 BT=1e4 之间 EB 差几百 nat，交换接受率 ≈ 0，阶梯必然停滞。
    p = _ladder(Ts=[1, 1e4])
    n_annealed = p.mcmc(tracefn='/tmp/test_ladder_anneal_trace.dat',
                        progressfn='/tmp/test_ladder_anneal_progress.dat',
                        burnin=100, thin=2, samples=30,
                        progress=False, anneal_after=40, anneal_factor=3.)
    check_above("停滞退火触发次数", n_annealed, 0)
    check("退火后阶梯归位违例数",
          sum(0 if abs(t.BT - BT) < 1e-12 else 1
              for BT, t in p.trees.items()), 0)
    with open('/tmp/test_ladder_anneal_trace.dat') as f:
        lines = f.readlines()
    check("退火后 trace 仍完整", len(lines), 30)
    check("退火后记录字段数", len(json.loads(lines[-1])), 6)


if __name__ == '__main__':
    for name in sorted(dir()):
        if name.startswith('test_'):
            globals()[name]()
    print("\n实验记录输出完毕（pytest 下用 -s 查看）。")
