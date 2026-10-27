# PIJit 原理入门

这篇文档给没接触过 PIJit 的读者补背景，后面几篇设计文档都建立在这里的概念上。内容基于 MindSpore r2.3 时期（2024 年 3 月）的代码，路径都相对于 `mindspore/ccsrc/pipeline/jit/pi/`。

## 1. 为什么需要 PIJit

MindSpore 有两种执行方式：

- **静态图（`PSJit`，GRAPH_MODE）**：解析 Python 源码的 AST，翻译成 MindSpore IR 再整体编译。性能好，但只支持 Python 语法的一个子集，写法稍微“动态”一点就报错。
- **动态图（PYNATIVE_MODE）**：逐条执行 Python，灵活但慢。

PIJit 走中间路线，用法是 `jit(fn=func, mode="PIJit")`。它在 **Python 字节码** 层面工作：能编译进图的部分编译成图，编译不了的部分交回解释器执行。用户不需要为了能编译而改写代码。这和 PyTorch 2.x 的 TorchDynamo 思路一致。

## 2. 入口：PEP 523 帧求值钩子

`common.cc` 用 `_PyInterpreterState_SetEvalFrameFunc` 把 CPython 的帧求值函数替换成 `pijit::EvalFrame`。一个被标记的函数被调用时：

1. 先查 **Guard 缓存**（`graph_guard/`）：之前编译过、且这次的输入满足编译时记录的 Guard 条件，就直接复用编译结果。
2. 否则进入编译流程，生成一个新的 Python 代码对象，替换掉原函数的 `f_code` 再执行。

## 3. 编译流程

```mermaid
flowchart LR
    BC["原始字节码"] --> GB["GraphBuilder<br/>graph_capture/graph_build.cc"]
    GB --> GA["GraphAnalyzer<br/>graph_capture/graph_analyzer.cc"]
    GA --> BI["BytecodeInliner<br/>graph_capture/bytecode_inliner.cc"]
    BI --> CG["CodeBreakGenerator<br/>graph_capture/code_generator.cc"]
    CG --> NEW["新字节码<br/>调用图 + 解释执行剩余部分"]
```

### 3.1 GraphBuilder：符号执行字节码

`GraphBuilder::TraceRun` 从第一条指令开始，用一个模拟的帧（`FrameStates`：局部变量、操作数栈、闭包）逐条“执行”字节码。每条指令由 `DoByteCode` 分发到对应的处理函数，例如 `DoItemAccess` 处理 `BINARY_SUBSCR` / `STORE_SUBSCR`，`DoGlobalAccess` 处理 `LOAD_GLOBAL` / `STORE_GLOBAL`。

执行过程中产生两类数据：

- **ValueNode 图**：每个值是一个 `ValueNode`，记录产生它的 opcode、输入节点、所在 bci。函数调用是 `CallNode`，可以带一个被内联的子图。
- **AObject 抽象类型**（`abstract_object.cc`）：每个 `ValueNode` 挂一个 `AObject`，描述“这是什么类型的对象、知不知道具体值、元素是什么类型”。例如 `AbstractTuple` 记录每个元素的 `AObject`。类型推导做得越细，后面能入图的东西越多。

某条指令处理不了时，`StopTraceAt(bci, reason)` 记录断点，追踪停止。

### 3.2 GraphAnalyzer：决定哪些节点入图

`GraphAnalyzer::Analyze` 从返回值和断点处的存活变量往回看，把节点分成两类：

- **captured**：能交给 MindSpore 编译的节点，主要是 Tensor 运算、Primitive、Cell 调用。
- **escaped**：必须由 Python 解释器执行的节点，比如调用未知的 Python 函数。

分析结束时还会决定 `need_interpret_`：如果整个函数都能入图，并且参数、返回值都是图支持的类型，就可以直接调用图，完全不生成 Python 包装代码。

### 3.3 CodeBreakGenerator：生成新字节码

`CodeBreakGenerator::MakeCode` 生成替换原函数的新代码对象，大致顺序是：

1. `Build()`：生成解释执行部分（escaped 节点）的字节码；
2. `CallCapturedCode` + `FixInterpretOuput`：把 captured 部分包装成名为 `<pijit.compile>` 的函数并调用，再把图的输出拆包写回局部变量；
3. （我新增的）回放副作用：`SideEffect::RestoreSideEffect`；
4. `CallUntrackedCode`：断点之后的原始字节码被打包成名为 `<pijit.resume>` 的新函数并调用，`BreakAtIf` / `BreakAtBlock` 分别处理断在 if 分支、with/try/for 代码块上的情况；
5. `MakeReturn`。

### 3.4 断图次数

每发生一次断图，`get_code_extra(func)["break_count_"]` 加一。`tests/st/pi_jit/` 下的很多用例就是用 `break_count_ == 0` 判断一个特性是否被完整支持，我写的副作用测试也沿用了这个做法。

## 4. 我的改动落在哪里

| 阶段 | 文件 | 我的改动 |
| --- | --- | --- |
| 抽象类型 | `abstract_object.cc` | 元组切片、列表乘法的类型推导 |
| 构图 | `graph_build.cc` | `DoSideEffect`、`STORE_SUBSCR` 分支、`LIST_EXTEND` 常量展开、循环头停止追踪、放宽 `ReplaceAll` |
| 构图 | `node.cc` | `store_attr` / `store_subscr` 不再直接改写共享的抽象对象 |
| 分析 | `graph_analyzer.cc` | `HandleCallableToGraph`、`HandleSideEffectNodeForCapture`、断图后清理副作用 |
| 内联 | `bytecode_inliner.cc` | 子图副作用合并、冗余 `STORE_DEREF` 消除 |
| 代码生成 | `code_generator.cc` | `FOR_ITER` 断点修复、kw-only 参数、副作用存活节点与回放 |
| 新模块 | `side_effect.h/.cc` | `SideEffect` 类 |
| 测试 | `tests/st/pi_jit/control_flow/test_side_effect.py` | 8 个副作用用例 |
