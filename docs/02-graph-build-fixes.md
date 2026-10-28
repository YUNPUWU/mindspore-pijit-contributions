# 02 Fixes in Graph Building and Code Generation

This document covers three patches:

| Commit | Date | PR | Patch |
| --- | --- | --- | --- |
| `8f94ed7` add loop trace switch and store subscr default crack map | 2023-12-30 | !63765 | [0003](../patches/02-graph-build-fixes/0003-loop-trace-switch-and-store-subscr-tracking.patch) |
| `b053d50` fix LIST EXTEND failed and kwonlyargs bug | 2024-01-05 | !63896 | [0004](../patches/02-graph-build-fixes/0004-fix-LIST_EXTEND-and-kwonlyargs.patch) |
| `584a519` fix listextend bugs | 2024-01-06 | !63919 | [0005](../patches/02-graph-build-fixes/0005-fix-listextend-bugs.patch) |

PR !63765 is titled "refactor codegen and support loop trace by chaiyouheng". I opened it, and it also contains my teammate chaiyouheng's code-generation refactor. The sections below only cover my own commit, `8f94ed7`.

---

## 1. Stop tracing at loop heads (`graph_build.cc`)

I added a check to the main loop of `GraphBuilder::TraceRun`: before executing each instruction, it looks at whether the current position is a loop head:

```cpp
Block *loop_head = graph_->GetCFG()->GetBlockByBci(cur_bci_);
if (loop_head->is_loop_head()) {
  graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceLoop_Unsupported);
  return StopTraceReason::kStopTraceLoop_Unsupported;
}
```

The purpose is to make the graph break **land on the loop boundary**, with an explicit stop reason: the code before the loop goes into the graph as usual, and the loop itself is handed to the interpreter. Later the community moved this check into `DoByteCode` and added a `kLoopUnrolling` option: with loop unrolling enabled, tracing no longer stops here.

## 2. Block boundaries when breaking at `FOR_ITER` (`code_generator.cc`)

When a graph break happens at a `FOR_ITER` that does not actually form a loop, `FindBlock` has to work out where that block ends. The original code was:

```cpp
if (list[start_bci]->op() == FOR_ITER && block_end == start_bci - 1) {
  block_end = list[block_end]->extra_jump()->bci();        // uses the instruction before FOR_ITER
}
```

At this point `block_end == start_bci - 1`, so `list[block_end]` is the instruction **before** `FOR_ITER`. It is usually not a jump, so `extra_jump()` may be null, and the computed range is wrong anyway. The right value is `FOR_ITER`'s own jump target (the loop exit) minus one:

```cpp
  block_end = list[start_bci]->extra_jump()->bci() - 1;
```

`FindBlock` has two implementations selected by Python version (`#if PY_MINOR_VERSION < 9` separates the bytecode before and after 3.9), and I applied the same fix to both.

## 3. `store_attr` / `store_subscr` no longer modify shared abstract objects (`node.cc`)

The original implementation modified the node's `AObject` directly on an attribute or subscript write:

```cpp
void ValueNode::store_subscr(ValueNode *sub, ValueNode *v) {
  // TODO(chaiyouheng): track variable modify, replace all node that represent the same object
  if (vobj_) {
    vobj_->SetItem(sub->vobj_, v->vobj_);
  }
  subscr_ = true;
}
```

The problem is that the same `AObject` can be referenced by several nodes at once (as the TODO in the original comment points out). A write through one node silently changes the type information of the others. I changed both functions to only set a flag:

```cpp
void ValueNode::store_subscr(ValueNode *sub, ValueNode *v) { subscr_ = true; }
```

`get_attr` / `binary_subscr` check the `attr_` / `subscr_` flags before reading; for an object that has been written to they return `kTypeAnyValue` directly instead of trusting the cached abstract information. This is conservative but correct. Precise tracking of modified objects was added later by the side-effect module (see [04](04-side-effect.md)), which replaces nodes instead.

## 4. Extracting `HandleCallableToGraph` (`graph_analyzer.cc`)

The part of `AddToCaptured` that decides whether a callable can go into the graph (Cells, Primitives, MetaFuncGraphs, functions marked `constexpr`, or objects carrying a MindSpore support flag) was extracted into its own `GraphAnalyzer::HandleCallableToGraph`. `AddToCaptured` becomes shorter, and the rule can be reused elsewhere. The same commit also removed stray semicolons after function bodies and an unused `guard_backup` variable, and added missing includes.

---

## 5. Python 3.9's `LIST_EXTEND <const tuple>` (`graph_build.cc`)

Starting with Python 3.9, constant list literals are compiled differently:

```
# [1, 2, 3]
BUILD_LIST     0
LOAD_CONST     (1, 2, 3)
LIST_EXTEND    1
```

After handling `LIST_APPEND` / `LIST_EXTEND`, `GraphBuilder::DoMergeOp` calls `ReplaceMergeOp` to merge it with the preceding `BUILD_LIST` into a new `BUILD_LIST` node. The original implementation only accepted `BUILD_LIST` / `BUILD_TUPLE` nodes as the argument of `LIST_EXTEND`. When the argument was a `LOAD_CONST`, the merge failed and the list stayed a `LIST_EXTEND` node. None of the logic that relies on the `BUILD_LIST` form could be used after that: for example, `DoSetItem` can only perform the functional element replacement on `BUILD_LIST` nodes, so the common pattern `x = [0, 0]; x[0] = t` broke the graph at the subscript write.

The fix expands the constant tuple into one `LOAD_CONST` node per element and merges them into a `BUILD_LIST`:

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

Expanding into element nodes, rather than keeping one constant for the whole tuple, keeps the representation consistent with `BUILD_LIST`: later logic such as `x[i] = ...` and `DoSetItem` assumes that the inputs of a `BUILD_LIST` are its elements.

The follow-up patch the next day, `584a519`, adapted the code after merging the upstream branch: `Graph`'s `allocator` had become the member function `allocator()`, and the patch also corrected the `item.ptr()` call. When the community later refactored `ReplaceMergeOp`, it kept the approach of expanding constant arguments into elements, using `IsConstantValue()` + `UnpackConstObject()`.

## 6. Keyword-only arguments (`code_generator.cc`)

When PIJit calls generated code, it passes arguments according to the `PackArgs` convention: the first `co_argcount + co_kwonlyargcount` slots of `f_localsplus` are packed into **a single tuple of positional arguments** (`PackArgs` in `common.cc`). However, when `CodeBreakGenerator::MakeCode` built the new code object, it copied the original function's argument information:

```cpp
code_gen.SetArgsInfo(co_->co_argcount, co_->co_kwonlyargcount);
```

So for a function like `def f(a, *, b)`, the new code object still declared `b` as keyword-only, which conflicts with the convention that everything is passed positionally. The fix folds keyword-only arguments into the positional ones:

```cpp
code_gen.SetArgsInfo(co_->co_argcount + co_->co_kwonlyargcount, 0);
```

Because keyword-only arguments come right after positional arguments in CPython's frame layout, this does not change any local-variable slot numbers, and `co_varnames` needs no adjustment. `MindCodeBreakGenerator::MakeCode` later adopted the same approach.
