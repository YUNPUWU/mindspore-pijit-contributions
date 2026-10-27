# 01 元组切片与列表乘法的类型推导

| 提交 | 日期 | PR | 补丁 |
| --- | --- | --- | --- |
| `63863e7` support tuple type derive | 2023-11-25 | !62405 | [0001](../patches/01-type-inference/0001-support-tuple-type-derive.patch) |
| `c9f43f9` support tuple slice type inference | 2023-12-12 | !63085 | [0002](../patches/01-type-inference/0002-support-tuple-slice-type-inference.patch) |

改动文件：`graph_capture/abstract_object.cc`（`AbstractTuple::GetItem`、`AbstractTuple::Binary`）。完整函数见 [excerpts/type_inference_abstract_object.cc](../excerpts/type_inference_abstract_object.cc)。

## 背景

`GraphBuilder` 每处理一条字节码，都要给结果算出一个抽象对象（`AObject`）。元组和列表都用 `AbstractTuple` 表示，它记录了每个元素的 `AObject`。只要元素类型已知，后续的 `t[0] + 1`、把元素传给某个算子等操作就能继续推导、继续入图。

改动之前，`AbstractTuple::GetItem` 只支持整数下标：

```cpp
Py_ssize_t index = GetTupleIndex(k, this->size());
if (index == -1) {
  return AObject::MakeAObject(kTypeAnyValue);   // 切片走到这里，结果变成“未知类型”
}
return items_[index];
```

所以 `t[::2]` 这样的切片结果只能推成 `kTypeAnyValue`，后面所有依赖它的节点都失去了类型信息。

## 第一版及其问题（!62405）

第一版给 `GetItem` 加上了对 `kTypeSlice` 下标的处理，用 `PySlice_Unpack` 解出 start / stop / step 再构造结果元组。问题在于它**替换**了原来的整数下标分支：非切片下标会返回一个空元组，而不是对应的元素。这是一个回归问题，维护者在 2023-11-29 通过 !62564（`Revert "support tuple type derive"`）把它回退了。

这次回退给我的教训是：在已有逻辑上加分支时，原有路径要保持不变，并且要有覆盖原路径的测试。

## 第二版：增量地加分支（!63085）

第二版只在下标是切片时进入新逻辑，整数下标的原路径一行没动：

```cpp
if (k->GetType() == AObject::kTypeSlice) {
  if (this->GetPyObject().ptr() != nullptr) {
    return this->AbstractSequence::GetItem(k);       // 有真实对象：交给通用实现按真实值计算
  }
  AObject *resultTuple = AObject::MakeAObject(this->type_);   // 保持 tuple / list 类型不变
  Py_ssize_t start, stop, step;
  if (PySlice_Unpack(slicePyObject, &start, &stop, &step) < 0) {
    return AObject::MakeAObject(kTypeAnyValue);
  }
  if (start >= stop) {
    return resultTuple;                               // 空切片
  }
  Py_ssize_t sliceLength = PySlice_AdjustIndices(this->items().size(), &start, &stop, step);
  if (start == 0 && step == 1 && sliceLength == this->size()) {
    return this;                                      // t[:]，直接复用
  }
  if (step > 1) { ... 按步长挑出元素 ... }
  return AObject::MakeAObject(kTypeAnyValue);         // 其他情况保守地退化为未知
}
// 下面是原来的整数下标逻辑，保持不变
```

设计上的几点考虑：

- **有真实值就用真实值。** 如果这个元组在追踪时有对应的 Python 对象（比如常量元组），通用实现 `AbstractSequence::GetItem` 可以直接切片得到精确结果，不需要自己推。新逻辑只用于“只知道元素类型、不知道具体值”的元组。
- **结果类型跟随原容器。** 用 `this->type_` 创建结果，所以列表切片的结果仍然是列表，元组切片的结果仍然是元组（第一版写死成了 `kTypeTuple`）。
- **用 CPython 的 `PySlice_AdjustIndices` 规范化下标**，处理负数下标、越界等情况，语义和 Python 一致。
- **整段切片返回 `this`**，避免为 `t[:]` 复制一份抽象对象。
- **不确定就退化。** 没有实现的情况一律返回 `kTypeAnyValue`，宁可少入图，也不推出错误的类型。

### 列表乘法

同一个 PR 里还给 `AbstractTuple::Binary` 加了 `list * int` 的推导：

```cpp
if (this->GetType() == kTypeList && op == BINARY_MULTIPLY && o != nullptr && o->GetType() == kTypeInt) {
  // [a, b] * 3  ->  [a, b, a, b, a, b]，逐个复制元素的抽象对象
}
```

这样 `[x] * n` 这种常见的初始化写法之后，每个元素的类型仍然已知。

## 示例

```python
@jit(mode="PIJit")
def f(a, b, c, d):          # a、b、c、d 是 Tensor
    t = (a, b, c, d)
    u = t[::2]              # 改动前：AnyValue；改动后：(Tensor, Tensor)
    return u[0] + u[1]      # 改动后可以继续入图
```

## 已知不足

- `step > 1` 的分支里游标从 0 开始，忽略了 `start`，`t[1::2]` 会选到错误的元素。正确写法是 `for (cursor = start; cursor < stop; cursor += step)`，或者直接用 `sliceLength` 计数。
- `step == 1` 但不是整段切片（如 `t[1:3]`）、负步长都没有实现，会退化为未知类型。
- `list * int` 没有检查整数的具体值是否已知。`o->GetPyObject()` 为空时，`PyLong_AsLong` 会返回 -1 并留下一个 Python 异常。应该先判空，取不到值就退化为未知类型。
