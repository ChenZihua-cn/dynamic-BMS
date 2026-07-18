# 2026-07-18: 测试案例严格化

## 概述
替换了5个测试文件中12个弱测试/作弊测试，替换为有实际断言能力的严格测试。`conda BMS` 环境下152个测试全部通过。

## 各文件变更

### test/test_core_basics.py
- **删除** `test_constants_mutability` — 测试Python dict可变性，与BMS逻辑无关
- **删除** `constants` 导入
- **替换** `test_canonical`（原断言：`len(can) > 0`）→ 拆为4个测试：
  - `test_canonical_param_rename` — 验证参数重命名（`_a0_` → `c1`）
  - `test_canonical_idempotent` — canonical(canonical(x)) == canonical(x)
  - `test_canonical_commutative` — `(x + _a0_)` 与 `(_a0_ + x)` 规范形式相同
  - `test_canonical_distinct` — 不同表达式规范形式不同

### test/test_mcmc.py
- **替换** `test_mcmc_step_no_data`（零断言）→ 每步断言size ∈ [1, max_size]、E/BIC有限、5步后结构变化
- **替换** `test_mcmc_step_with_data`（原：`assert t.size > 0`）→ 断言size边界、E/BIC有限、树演化、E增幅 ≤ 1.0
- **替换** `test_predict`（原：仅`isinstance`检查）→ 断言长度匹配输入、值全部有限、非恒定预测、不同长度测试数据

### test/test_visualization.py
- **替换** `test_trace_plots`（零断言）→ 断言50条痕迹样本、BIC/Energy全部有限、非零方差（链有探索）、保留PNG输出
- **替换** `test_pred_vs_actual_plot`（零断言）→ 断言预测长度匹配、值全部有限、与真实值相关系数 > 0.5、保留散点图

### test/test_synthetic.py
- **转换** 5个软WARN测试为硬断言：
  - `test_trig`：WARN → `assert test_r2 > 0.3`
  - `test_2var`：WARN → `assert test_r2 > 0.3`
  - `test_3var_interact`：WARN → `assert test_r2 > 0.15`
  - `test_3var_rational`：WARN → `assert test_r2 > 0.1`
  - `test_3var_mixed`：WARN → `assert test_r2 > 0.05`
- **修复** 导入：`from helpers import r2_score` → `from test.helpers import r2_score`（pytest下原导入失败）

### test/test_rejection_rate.py
- **收紧** 拒绝率阈值从 `< 90%` → `< 80%`（与5-D升级决策规则一致）

### test/README.md
- 更新各文件测试描述和数量
- 移除"作弊/弱测试"章节（全部已修复）

## 验证
- `conda run -n BMS python -m pytest test/ -x -v` → 152 passed, 0 failed
