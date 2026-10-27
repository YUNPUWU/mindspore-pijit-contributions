# 02 构图与代码生成的若干修复

本篇包含三个补丁：

| 提交 | 日期 | PR | 补丁 |
| --- | --- | --- | --- |
| `8f94ed7` add loop trace switch and store subscr default crack map | 2023-12-30 | !63765 | [0003](../patches/02-graph-build-fixes/0003-loop-trace-switch-and-store-subscr-tracking.patch) |
| `b053d50` fix LIST EXTEND failed and kwonlyargs bug | 2024-01-05 | !63896 | [0004](../patches/02-graph-build-fixes/0004-fix-LIST_EXTEND-and-kwonlyargs.patch) |
| `584a519` fix listextend bugs | 2024-01-06 | !63919 | [0005](../patches/02-graph-build-fixes/0005-fix-listextend-bugs.patch) |

!63765 这个 PR 的标题是 “refactor codegen and support loop trace by chaiyouheng”，由我发起，里面同时包含了同事 chaiyouheng 的代码生成重构。下文只讲我自己的提交 `8f94ed7`。

---

## 1. 在循环头停止追踪（`graph_build.cc`）

我在 `GraphBuilder::TraceRun` 的主循环里加了一个检查，执行每条字节码之前先看当前位置是不是循环头：

```cpp
Block *loop_head = graph_->GetCFG()->GetBlockByBci(cur_bci_);
if (loop_head->is_loop_head()) {
  graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceLoop_Unsupported);
  return StopTraceReason::kStopTraceLoop_Unsupported;
}
```

这样做的目的是让断图**落在循环边界上**，并带上明确的停止原因：循环之前的代码正常入图，循环本身交给解释器执行。后来社区把这个检查移进了 `DoByteCode`，并加了 `kLoopUnrolling` 配置项：开启循环展开时不再停止。

## 2. `FOR_ITER` 处断图时的块边界（`code_generator.cc`）

断图发生在 `FOR_ITER` 上、但它其实不构成循环时，`FindBlock` 要算出这个代码块在哪里结束。原代码是：

```cpp
if (list[start_bci]->op() == FOR_ITER && block_end == start_bci - 1) {
  block_end = list[block_end]->extra_jump()->bci();        // 用的是 FOR_ITER 前一条指令
}
```

此时 `block_end == start_bci - 1`，`list[block_end]` 是 `FOR_ITER` **前面**那条指令，它通常不是跳转指令，`extra_jump()` 可能为空，算出的范围也不对。应该用 `FOR_ITER` 自己的跳转目标（循环退出位置）减一：

```cpp
  block_end = list[start_bci]->extra_jump()->bci() - 1;
```

`FindBlock` 按 Python 版本有两套实现（用 `#if PY_MINOR_VERSION < 9` 区分 3.9 之前和之后的字节码），两处都做了同样的修复。

## 3. `store_attr` / `store_subscr` 不再改写共享的抽象对象（`node.cc`）

原实现会在属性或下标写入时直接改写节点上的 `AObject`：

```cpp
void ValueNode::store_subscr(ValueNode *sub, ValueNode *v) {
  // TODO(chaiyouheng): track variable modify, replace all node that represent the same object
  if (vobj_) {
    vobj_->SetItem(sub->vobj_, v->vobj_);
  }
  subscr_ = true;
}
```

问题是同一个 `AObject` 可能同时被多个节点引用（原注释里的 TODO 也说明了这一点）。在一个节点上写入，会悄悄改掉其他节点的类型信息。我把这两个函数改成只设置标记：

```cpp
void ValueNode::store_subscr(ValueNode *sub, ValueNode *v) { subscr_ = true; }
```

`get_attr` / `binary_subscr` 在读取前会先检查 `attr_` / `subscr_` 标记，被写过的对象直接返回 `kTypeAnyValue`，不再相信缓存的抽象信息。这是一个保守但正确的做法：对象被修改后的精确跟踪，后来由副作用模块（见 [04](04-side-effect.md)）用“替换节点”的方式完成。

## 4. 抽取 `HandleCallableToGraph`（`graph_analyzer.cc`）

`AddToCaptured` 里判断“这个可调用对象能不能入图”的一段逻辑（Cell、Primitive、MetaFuncGraph、带 `constexpr` 标记的函数，或带 MindSpore 支持标记的对象）被抽成独立的 `GraphAnalyzer::HandleCallableToGraph`，`AddToCaptured` 变短，这条规则也可以在别处复用。同一个提交还顺手清理了函数体后多余的分号、未使用的 `guard_backup` 变量，并补全了头文件。

---

## 5. Python 3.9 的 `LIST_EXTEND <常量元组>`（`graph_build.cc`）

从 Python 3.9 开始，常量列表字面量的编译方式变了：

```
# [1, 2, 3]
BUILD_LIST     0
LOAD_CONST     (1, 2, 3)
LIST_EXTEND    1
```

`GraphBuilder::DoMergeOp` 处理完 `LIST_APPEND` / `LIST_EXTEND` 后，会调用 `ReplaceMergeOp` 把它和前面的 `BUILD_LIST` 合并成一个新的 `BUILD_LIST` 节点。但原实现只接受 `BUILD_LIST` / `BUILD_TUPLE` 节点作为 `LIST_EXTEND` 的参数，参数是 `LOAD_CONST` 时合并失败，列表就一直以 `LIST_EXTEND` 节点的形式存在。后续依赖 `BUILD_LIST` 表示的逻辑都用不上了：例如 `DoSetItem` 只能对 `BUILD_LIST` 节点做函数式的元素替换，于是 `x = [0, 0]; x[0] = t` 这种常见写法会在下标写入处断图。

修复方法是把常量元组展开成逐个元素的 `LOAD_CONST` 节点，再合并成 `BUILD_LIST`：

```cpp
case LIST_EXTEND:
  if (arg->GetOpcode() == LOAD_CONST && arg->GetVobj() != nullptr &&
      arg->GetVobj()->GetType() == AObject::kTypeTuple) {
    for (auto item : arg->GetVobj()->GetPyObject()) {
      ValueNode *v = graph->allocator().NewNode<ValueNode>(AObject::Convert(item.ptr()), LOAD_CONST, -1,
                                                           std::vector<ValueNode *>());
      v->SetGraph(graph);
      inputs.push_back(v);
    }
    container->getInputs() = inputs;
    container->SetOpcode(BUILD_LIST);
    container->SetOparg(inputs.size());
    break;
  }
```

展开成元素节点而不是保留一个整体常量，是为了和 `BUILD_LIST` 的表示保持一致：后面的 `x[i] = ...`、`DoSetItem` 等逻辑都假设 `BUILD_LIST` 的输入就是各个元素。

第二天的补丁 `584a519` 是在合并上游分支之后做的适配：`Graph` 的 `allocator` 变成了成员函数 `allocator()`，并修正了 `item.ptr()` 的写法。后来社区重构 `ReplaceMergeOp` 时，用 `IsConstantValue()` + `UnpackConstObject()` 延续了“把常量参数展开成元素”的处理。

## 6. keyword-only 参数（`code_generator.cc`）

PIJit 调用生成代码时，会按 `PackArgs` 的约定传参：`f_localsplus` 中前 `co_argcount + co_kwonlyargcount` 个槽位被打包成**一个位置参数元组**（`common.cc` 中的 `PackArgs`）。但 `CodeBreakGenerator::MakeCode` 生成新代码对象时，照搬了原函数的参数信息：

```cpp
code_gen.SetArgsInfo(co_->co_argcount, co_->co_kwonlyargcount);
```

于是对 `def f(a, *, b)` 这样的函数，新代码对象仍然声明 `b` 是 keyword-only 参数，和“全部按位置传入”的约定冲突。修复方式是把 kw-only 参数并入位置参数：

```cpp
code_gen.SetArgsInfo(co_->co_argcount + co_->co_kwonlyargcount, 0);
```

由于 CPython 的帧布局中 kw-only 参数紧跟在位置参数之后，这样改不影响局部变量的槽位编号，`co_varnames` 也不需要调整。`MindCodeBreakGenerator::MakeCode` 后来也采用了同样的写法。
