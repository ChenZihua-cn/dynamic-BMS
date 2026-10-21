# 先验矩匹配返回值的覆盖 bug（重构实现）

- 日期：2026-10-20
- 范围：**本仓库重构实现**，不是 BMS 2020 论文原版代码
- 文件：`dynamic-BMS/prior/fit.py`
- 函数：`fit_moment_matched_prior`（239--321 行）
- 重点：303--316 行，尤其是 314 与 316 行

## 0. 一句话结论

`fit_moment_matched_prior` 的迭代循环确实在更新 `avec` 并记录失配诊断，但 314 行用 `avec` 组装出的返回值在 316 行又被初值字典 `a` 覆盖。函数最终返回的 `Nopi_*` 与 `fit_unigram_prior` 的初值逐位相同；二阶矩 `Nopi2_*` 从未参与计算。

## 1. 现象

当前代码（`dynamic-BMS/prior/fit.py:300--316`）：

```python
    diag = {'ess': [], 'mean_size': [], 'mismatch': []}
    avec = np.asarray([a[o] for o in ops], dtype=float)
    for _ in range(n_iter):
        logw = -(N @ avec) - logq
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        model = w @ N
        mismatch = float(np.linalg.norm(model - tvec))
        diag['ess'].append(float(1.0 / np.sum(w ** 2)))
        diag['mean_size'].append(float(w @ sizes))
        diag['mismatch'].append(mismatch)
        avec = avec + eta * (model - tvec)

    out = dict([('Nopi_%s' % o, float(avec[i])) for i, o in enumerate(ops)])

    out = dict([('Nopi_%s' % o, a[o]) for o in ops])
```

关键点：

1. 301 行 `avec` 由初值 `a` 建立；`a` 在 278--282 行来自 `fit_unigram_prior`（或调用者传入的 `init`）。
2. 302--312 行的循环用重要性权重 `w` 计算模型侧一阶矩 `model = w @ N`，按 `avec += eta * (model - tvec)` 更新。
3. 308 行把每轮失配写入 `diag['mismatch']`。所以诊断会显示失配下降，循环本身不是空转。
4. 314 行把迭代后的 `avec` 组装成 `out`。
5. 316 行立刻用初值 `a` 重新组装 `out`，覆盖 314 行的结果。函数返回的是初值，不是迭代值。

## 2. 复现

在仓库根目录用项目虚拟环境运行：

```bash
cd dynamic-BMS
.venv/bin/python - <<'PY'
from prior import fit_unigram_prior, fit_moment_matched_prior

corpus = ['_a_*sqrt(L/g)', 'm*g*h', 'p*V/(n*R*T)', 'sin(x)/cos(y)']
unigram = fit_unigram_prior(corpus)
matched, diag = fit_moment_matched_prior(
    corpus, n_samples=200, n_iter=5, seed=0, return_diagnostics=True,
)

print('returned == unigram:', matched == unigram)
print('mismatch first/last:', diag['mismatch'][0], diag['mismatch'][-1])
print('returned:', matched)
PY
```

本机实测输出（2026-10-20）：

```
returned == unigram: True
mismatch first/last: 3.299351741210397 2.9717101278674556
returned: {'Nopi_sin': 2.639057329615259, ..., 'Nopi_**': 3.7376696182833684}
```

失配从 3.299 降到 2.972，说明迭代有效；但返回值与 `fit_unigram_prior` 完全相等，说明 316 行的覆盖生效。

现有测试 `test/test_prior.py::test_moment_matching_reduces_mismatch` 只断言 `diag['mismatch'][-1] < diag['mismatch'][0]`，不检查返回值，因此无法发现这个 bug。

## 3. 与论文原版的区别

BMS 2020 原版的先验拟合在 `upstream-bms/Prior/fit_prior.py`：

- 168--191 行的主循环：每轮新建 `Tree(prior_par=ppar)`，反复 `mcmc_step()`，同时累计
  `current['Nopi_%s'] += nopi` 与 `current['Nopi2_%s'] += nopi * nopi`；
- 203--209 行调用 `update_ppar`；
- 71--107 行的 `update_ppar` 对 `current` 里的**所有**项更新，包括 `Nopi_*`（α）与
  `Nopi2_*`（β）；二次项最后被截断到不小于 0。

也就是说，原版同时匹配一阶矩与二阶矩：

- α 对应 `⟨n_o⟩`；
- β 对应 `⟨n_o²⟩`（论文 Eq. (14)(15) 的二次项）。

当前重构实现的 `fit_moment_matched_prior`：

- 只构造一阶计数矩阵 `N`（每行是一棵树的 `n_o` 向量），没有 `n_o²` 的矩阵；
- 只更新 `Nopi_*`，docstring 也明确写“调整 `a_o` 使 `<n_o>` 与语料一致”；
- 没有 `Nopi2_*` 的输出，也没有 β 的更新；
- 再加上 316 行的覆盖，连一阶矩的迭代结果也没有返回。

因此它更准确的名称是“一阶矩重要性采样校准（且返回值被覆盖）”，不是完整的矩匹配。

## 4. 影响

1. **调用方拿到的是 unigram 初值。** 任何直接使用 `fit_moment_matched_prior` 返回值写先验文件或传入 `Tree` 的代码，实际使用的是 `fit_unigram_prior` 的结果，而不是迭代后的值。
2. **二阶矩完全缺失。** 即使修掉 316 行，当前函数也只匹配 `⟨n_o⟩`；它不会产生 `Nopi2_*`，因此不能复现 BMS 2020 的二次先验。对算子频率涨落的约束（β）没有进入目标函数。
3. **诊断与返回值不一致。** `diag['mismatch']` 下降会让人误以为返回的参数已经校准；实际返回值的失配仍是初值失配。这比“函数明显没实现”更隐蔽。
4. **测试无法兜底。** 现有测试只检查诊断，未检查返回值与初值的差异，也未检查 `Nopi2_*` 是否存在。
5. **影响范围限于显式调用。** `prior/registry.py` 的默认表 `bms_wiki_nv13_np13` 是一组静态数值，不调用 `fit_moment_matched_prior`；仓库内直接调用该函数的地方是 `prior/README.md` 的示例与 `test/test_prior.py`。因此既有实验若使用默认先验表，先验数值不受本 bug 影响；但任何按 README 重新校准并保存先验的流程，都会把 unigram 初值当成矩匹配结果写出去。

## 5. 修复建议

### 5.1 最小修复：不要覆盖返回值

删除 316 行，保留 314 行的 `out`：

```python
    out = dict([('Nopi_%s' % o, float(avec[i])) for i, o in enumerate(ops)])
```

并加一个回归测试：

```python
def test_moment_matching_returns_iterated_values():
    corpus = ['_a_*sqrt(L/g)', 'm*g*h', 'p*V/(n*R*T)', 'sin(x)/cos(y)']
    unigram = fit_unigram_prior(corpus)
    matched, diag = fit_moment_matched_prior(
        corpus, n_samples=4000, n_iter=40, eta=0.3, seed=1,
        return_diagnostics=True,
    )
    assert diag['mismatch'][-1] < diag['mismatch'][0]
    assert any(abs(matched[k] - unigram[k]) > 1e-8 for k in unigram)
```

### 5.2 完整修复：同时匹配二阶矩

如果函数要对应论文 Eq. (14)(15)，需要：

1. 在采样阶段同时累计 `N2[i, o] = n_o²`；
2. 目标向量增加 `tvec2[o] = ⟨n_o²⟩`（`corpus_moments` 已经能给出二阶矩）；
3. 用与一阶项相同的权重计算 `model2 = w @ N2`，更新 `b_o += eta * (model2 - tvec2)`；
4. 输出 `Nopi2_*`，并按原版把 `b_o` 截断到不小于 0；
5. 诊断里同时记录一阶与二阶失配。

### 5.3 接口层面的防呆

- 在返回值里附带 `diag['returned_from']`（`'iterated'` / `'init'`）或在文档中明确“只匹配一阶矩”，避免调用方误以为返回的是完整 BMS 先验。
- 如果保留 `init` 参数，应在返回前断言 `out` 与 `init` 不同（除非 `n_iter == 0`），否则容易再次出现“迭代了但返回初值”的静默失败。

## 6. 边界声明

这是本仓库重构实现中的问题，不是 BMS 2020 论文原版代码的问题。原版 `upstream-bms/Prior/fit_prior.py` 的 71--107 行与 168--210 行同时更新 α 与 β；本书第 2 章按原版讲解，未把本 bug 写进书稿。修复本 bug 不应修改 `upstream-bms/`，也不应把重构实现的行为当成论文结论。
