# Fix: `build_from_string` 丢失 `max_size` 参数

**日期**: 2026-07-17

## 问题

`test_regression.py` 的 viz 测试（line 436-499）生成的 `test_viz_trace.png` 和 `test_viz_pred_vs_actual.png` 图片上显示的公式严重过拟合，为 200+ 字符的怪物表达式（cosh, tan, sinh, abs, exp, fac 等大量非线性操作符嵌套），完全无法反映 `y = 2x + 1` 的线性真值。

## 根因

`core/tree_base.py` 的 `build_from_string()` 内部调用 `self.__init__()` 时漏传 `max_size` 参数：

```python
# tree_base.py:285 (修复前)
self.__init__(ops=self.ops, prior_par=self.prior_par,
              x=self.x, y=self.y, BT=self.BT, PT=self.PT,
              parameters=parameters, variables=variables)
```

`__init__` 中 `max_size` 默认值为 50，因此用户传入的 `max_size=10` 被覆盖为 50。MCMC 可以在 50 节点的搜索空间内自由膨胀，找到 BIC 极小但物理意义全无的过拟合表达式。

## 修复

1. **`core/tree_base.py:285`**：`__init__` 调用增加 `max_size=self.max_size`，正确传递用户在构造函数中指定的 size 上限
2. **`test_regression.py:451`**：viz 测试初始化从 `(x0 + _a0_)` 改为 `((x0 * _a0_) + _a1_)`，使模型能表达 `y = a*x + b` 的线性形式（原初始化斜率固定为 1，无法拟合 `y = 2x + 1`）

## 效果

| | 修复前 | 修复后 |
|---|---|---|
| 公式 | 222 字符嵌套怪物 | `((_a0_ * tan(_a1_)) + (tan(_a1_) * x0))` ≈ `1.99x + 0.98` |
| BIC | -3.28 | -2.59 |
| Tree size | 50 节点 | 10 节点（受限） |
| 参数 | 不可解释 | _a0_ ≈ 0.49, _a1_ ≈ -2.04 → tan(_a1_) ≈ 1.99（斜率） |
