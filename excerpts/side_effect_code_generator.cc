// Excerpt from MindSpore (https://github.com/mindspore-ai/mindspore)
// Copyright Huawei Technologies Co., Ltd. Licensed under the Apache License, Version 2.0.
// Snapshot: commit a8005072ba60ddeafffa84ea3d6953f95db179d0 (2024-03-26, merge of PR !67269)
// 内容: 代码生成阶段：把副作用存活节点作为图输出，图执行完后调用 SideEffect::RestoreSideEffect 回放
// 说明: 这里是包含我改动的完整函数体，便于阅读上下文；函数中并非每一行都由我编写，
//       逐行归属请以 patches/ 目录下的原始补丁为准。

// ---- mindspore/ccsrc/pipeline/jit/pi/graph_capture/code_generator.cc  L992-L1051 ----
py::object CodeBreakGenerator::MakeCode(bool make_graph, Graph *graph) {
  auto jcr = getJitCompileResults(reinterpret_cast<PyObject *>(co_), false);

  if (make_graph) {
    // all parameters is graph supported
    captured_.inputs.clear();
    captured_.outputs.clear();
    interpret_.operations = std::move(captured_.operations);
  }

  CodeGenerator code_gen(&interpret_);
  code_gen.SetGlobals(GetGlobals());
  code_gen.Init();
  for (auto i : captured_.inputs) {
    code_gen.MarkAlive(i);
  }
  code_gen.Build();

  CallCapturedCode(&code_gen);
  FixInterpretOuput(&code_gen);
  // ... handle side effects
  if (make_graph == false) {
    graph->GetSideEffect()->RestoreSideEffect(&code_gen);
    auto vec = graph->GetSideEffect()->CollectSideEffectAliveNodes();
    interpret_.outputs.resize(interpret_.outputs.size() - vec.size());
  }

  CallUntrackedCode(&code_gen);
  MakeReturn(&code_gen);

  std::string co_name = PyUnicode_AsUTF8(co_->co_name);
  if (make_graph) {
    co_name = MakeCompiledName(co_name);
  }
  co_name = std::to_string(jcr->IncCodeCount()) + "R." + co_name;

  int nlocals = static_cast<int>(code_gen.GetLocalsMap().size());
  nlocals = std::max(nlocals, co_->co_nlocals);
  nlocals = std::max(nlocals, cfg_->GetLocalCount());

  code_gen.SetArgsInfo(co_->co_argcount + co_->co_kwonlyargcount, 0);
  code_gen.SetLocalsCount(nlocals);
  code_gen.SetCodeFlags(co_->co_flags);
  code_gen.SetFirstLineNumber(co_->co_firstlineno);
  code_gen.SetVariableNames(py::cast<std::vector<std::string>>(co_->co_varnames));
  code_gen.SetCellVariableNames(py::cast<std::vector<std::string>>(co_->co_cellvars));
  code_gen.SetFreeVariableNames(py::cast<std::vector<std::string>>(co_->co_freevars));
  code_gen.SetCodeName(co_name);
  code_gen.SetFileName(py::reinterpret_borrow<py::object>(co_->co_filename));

  code_gen.EraseUnusedInstr();
  py::object result = CodeGenerator::Transform(code_gen.GetCode());
  if (make_graph) {
    JitCompileResults *child = getJitCompileResults(result.ptr());
    child->stat = CodeExtra::GRAPH_CAPTURED;
    child->conf = jcr->conf;
    child->tbs = jcr->tbs;
  }
  return result;
}

// ---- mindspore/ccsrc/pipeline/jit/pi/graph_capture/code_generator.cc  L1090-L1103 ----
void CodeBreakGenerator::Init(const Graph *graph, const GraphAnalyzer::CapturedInfo *info) {
  break_bci_ = graph->GetStopTraceBci();
  cfg_ = graph->GetCFG().get();
  std::vector<ValueNode *> alive_nodes = graph->CollectAliveNode(break_bci_, &alive_locals_);

  interpret_.inputs = graph->GetFrame(0).GetLocals();
  interpret_.outputs = std::move(alive_nodes);
  interpret_.operations = info->ordered_escaped_locals;
  auto vec = graph->GetSideEffect()->CollectSideEffectAliveNodes();
  interpret_.outputs.insert(interpret_.outputs.end(), vec.begin(), vec.end());
  captured_.inputs = std::vector<ValueNode *>(info->captured_locals.inputs.begin(), info->captured_locals.inputs.end());
  captured_.outputs = CollectGraphOutputs(info->escaped_locals, interpret_.outputs);
  captured_.operations = info->captured_locals.order;
}
