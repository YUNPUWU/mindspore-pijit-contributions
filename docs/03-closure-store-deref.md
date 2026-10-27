# 03 消除冗余的 `STORE_DEREF`

| 提交 | 日期 | PR | 补丁 |
| --- | --- | --- | --- |
| `38440d6` elimate store deref | 2024-02-02 | !65004 | [0006](../patches/03-closure-optimization/0006-eliminate-redundant-store-deref.patch) |

改动文件：`graph_capture/bytecode_inliner.cc`（`BytecodeInliner::EliminateClosureSideEffect`）。完整函数见 [excerpts/side_effect_bytecode_inliner.cc](../excerpts/side_effect_bytecode_inliner.cc)。

## 背景

`BytecodeInliner` 会把能内联的函数调用展开到调用方，再把整个函数重新生成为一份字节码。如果函数有闭包变量（`co_cellvars` / `co_freevars` 不为空），`STORE_DEREF` / `DELETE_DEREF` 就是对闭包单元（cell）的写操作，属于**闭包副作用**：这类写入对外部可见，必须保留并由解释器执行。

`EliminateClosureSideEffect` 原本只做一件事：如果某个闭包写入之后不会再被任何存活的闭包读取，就把它删掉。

## 问题

内联和字节码重建之后，可能出现“把从闭包读出的值原样写回同一个闭包变量”的节点：

```
LOAD_DEREF   i
STORE_DEREF  i        # 等价于 cell_i = cell_i，没有任何效果
```

它在语义上什么都没做，但仍然会被当成闭包副作用保留下来，让本来可以省掉的写操作留在生成的字节码里。

## 实现

在原有的删除逻辑之后，再遍历一次已追踪节点，删除满足以下三个条件的 `STORE_DEREF`：

```cpp
for (auto item = traced_nodes_.begin(); item != traced_nodes_.end();) {
  if ((*item)->GetOpcode() == STORE_DEREF &&
      (*item)->getInputs()[0]->GetOpcode() == LOAD_DEREF &&            // 1. 写入的值直接来自 LOAD_DEREF
      (*item)->getInputs()[0]->GetOparg() == (*item)->GetOparg() &&    // 2. 读和写是同一个闭包槽位
      (*item)->getInputs()[0]->GetGraph() == (*item)->GetGraph()) {    // 3. 读和写属于同一个（子）图
    item = traced_nodes_.erase(item);
  } else {
    ++item;
  }
}
```

第 3 个条件是这里最容易漏掉的地方。`LOAD_DEREF` / `STORE_DEREF` 的 oparg 是**相对于各自代码对象**的 cell / free 变量下标。内联之后，同一个 `traced_nodes_` 列表里混有来自不同函数的节点，两个节点 oparg 相同，并不代表它们指向同一个闭包单元。只有在同一个图里，oparg 相同才说明是同一个变量。

遍历时用 `erase` 的返回值继续迭代，避免在删除元素后使用失效的迭代器。
