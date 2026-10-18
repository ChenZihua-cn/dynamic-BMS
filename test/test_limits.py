"""极限行为：x → 0 时模型必须保持有限的渐近直觉。

物理动机 —— 若已知观测量在某个极限下必须保持有限（如恒温热力学
极限下体积不应发散），则含该变量于分母的表达式是可疑的。

实验对象：AsymptoticPrior 议会专家，对"除以 must-be-finite 变量"
施加结构惩罚；以及 Boyle 定律 V = const/P 在双专家 MCMC 下的恢复
（即使软先验与似然 mildly 失配，数据似然仍应主导链走向正确形式）。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from core import Tree
from experts.parliament.occam import OccamPrior
from experts.parliament.asymptotic import AsymptoticPrior
from test.lab import check, check_below, measure, reproducible_state, section


def test_asymptotic_penalty_rules():
    """基本惩罚规则：除以声明为 must-be-finite 的变量 → +0.5/处，
    其余情形 → 0；多处违规叠加。"""
    section("极限行为：基本惩罚规则")
    expert = AsymptoticPrior(finite_at_zero=['x', 'y'])

    t_nodiv = Tree(variables=['x'], parameters=['a'], from_string='(_a0_ * x)')
    check("无除法 → 0", expert.evaluate_structural(t_nodiv), 0.0)

    t_div = Tree(variables=['x', 'y'], parameters=['a'], from_string='(x / y)')
    check("除以 must-be-finite 变量 → 0.5",
          expert.evaluate_structural(t_div), 0.5)

    t_param = Tree(variables=['x'], parameters=['a'], from_string='(x / _a0_)')
    check("参数在分母（非 must-be-finite）→ 0",
          expert.evaluate_structural(t_param), 0.0)

    t_stack = Tree(variables=['x', 'y', 'z'], parameters=['a'],
                   from_string='((x / y) + (z / y))')
    check("两处违规叠加 → 1.0", expert.evaluate_structural(t_stack), 1.0)

    t_out = Tree(variables=['x', 'y'], parameters=['a'], from_string='(x / y)')
    expert_x = AsymptoticPrior(finite_at_zero=['x'])
    check("未声明的变量在分母 → 0", expert_x.evaluate_structural(t_out), 0.0)


def test_nested_denominators():
    """嵌套分母：目标变量藏在复合分母表达式中仍被识别。"""
    section("极限行为：嵌套分母")
    expert = AsymptoticPrior(finite_at_zero=['y'])

    t_nested = Tree(variables=['x', 'y'], parameters=['a'],
                    from_string='(x / (_a0_ + y))')
    check("一层嵌套分母 → 0.5", expert.evaluate_structural(t_nested), 0.5)

    t_deep = Tree(variables=['x', 'y'], parameters=['a', 'b'],
                  from_string='(x / (_a0_ * (_b0_ + y)))')
    check("深层嵌套分母 → 0.5", expert.evaluate_structural(t_deep), 0.5)


def test_no_opinion_when_undeclared():
    """空声明：finite_at_zero 为 None 或 [] 时专家不表态（恒 0）。"""
    section("极限行为：空声明")
    t = Tree(variables=['x', 'y'], parameters=['a'], from_string='(x / y)')
    check("None → 不表态",
          AsymptoticPrior(finite_at_zero=None).evaluate_structural(t), 0.0)
    check("[] → 不表态",
          AsymptoticPrior(finite_at_zero=[]).evaluate_structural(t), 0.0)


def test_boyle_law_penalty_pattern():
    """Boyle 定律的渐近排布：V = const/P 中 P 在分母被判可疑。

    物理设定：恒温热力学极限 P → 0 时 V 必须发散——若理论要求 V 有限，
    则 P 出现于分母是渐近不匹配。四个排布逐一检验。
    """
    section("极限行为：Boyle 定律排布")
    t_denom = Tree(variables=['P'], parameters=['a'], from_string='(_a0_ / P)')
    check("P 在分母 → 0.5",
          AsymptoticPrior(finite_at_zero=['P']).evaluate_structural(t_denom),
          0.5)

    t_mul = Tree(variables=['P'], parameters=['a'], from_string='(_a0_ * P)')
    check("无除法 → 0（P→0 时趋于 0，有限）",
          AsymptoticPrior(finite_at_zero=['P']).evaluate_structural(t_mul),
          0.0)

    t_pv = Tree(variables=['P', 'V'], parameters=['a'], from_string='(P / V)')
    check("V 在分母（未声明）→ 0",
          AsymptoticPrior(finite_at_zero=['P']).evaluate_structural(t_pv),
          0.0)

    t_num = Tree(variables=['P'], parameters=['a'], from_string='(P / _a0_)')
    check("P 仅在分子 → 0",
          AsymptoticPrior(finite_at_zero=['P']).evaluate_structural(t_num),
          0.0)


def test_boyle_law_mcmc_recovery():
    """Boyle 定律双专家恢复：软先验与似然 mildly 失配时，似然仍主导。

    生成 V = const/P 含噪数据，以 OccamPrior + AsymptoticPrior 启动
    MCMC。渐近专家惩罚 P/分母（此处与真值失配），但 50 步内链应保持
    有限能量，且结构能 = 两专家原始惩罚的加权和。
    """
    section("极限行为：Boyle 定律 MCMC 恢复")
    reproducible_state(42)
    const_true = 2.0
    P_vals = np.linspace(0.5, 3.0, 40)
    V_true = const_true / P_vals
    V_obs = V_true + np.random.normal(0, 0.05, 40)

    x_df = pd.DataFrame({'P': P_vals})
    y_series = pd.Series(V_obs, name='V')

    t = Tree(
        variables=['P'], parameters=['a', 'b'],
        x=x_df, y=y_series,
        parliaments=[
            OccamPrior(),
            AsymptoticPrior(finite_at_zero=['P']),
        ],
        from_string='((_a0_ / P) + _b0_)',
        max_size=20,
    )

    check("双专家注册", set(t.parliament_weights.keys()),
          {'occam', 'asymptotic'})
    w = t.parliament_weights
    check("权重归一", sum(w.values()), 1.0, tol=1e-10)

    structural = t._parliament_energy_structural()
    expected = 0.5 * 5 + 0.5 * 0.5
    measure("结构能（期望 2.75 = 0.5×size5 + 0.5×0.5）", structural)
    check("结构能 = 加权和", structural, expected, tol=1e-10)

    for i in range(50):
        t.mcmc_step(verbose=False)
        check("E 有限", np.isfinite(float(t.E)), True, context=f"step {i}")
        check("EP 有限", np.isfinite(float(t.EP)), True, context=f"step {i}")

    check_below("议会结构缓存规模", len(t._parliament_structural_cache), 10000)
    check("末态 E 有限", np.isfinite(float(t.E)), True)
    check("末态 EP 有限", np.isfinite(float(t.EP)), True)


if __name__ == '__main__':
    for name in sorted(dir()):
        if name.startswith('test_'):
            globals()[name]()
    print("\n实验记录输出完毕（pytest 下用 -s 查看）。")
