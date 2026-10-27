// Excerpt from MindSpore (https://github.com/mindspore-ai/mindspore)
// Copyright Huawei Technologies Co., Ltd. Licensed under the Apache License, Version 2.0.
// Snapshot: commit a8005072ba60ddeafffa84ea3d6953f95db179d0 (2024-03-26, merge of PR !67269)
// 内容: 图分析阶段：可入图调用判定、副作用节点强制入图、graph break 后清理副作用、强制生成解释执行代码
// 说明: 这里是包含我改动的完整函数体，便于阅读上下文；函数中并非每一行都由我编写，
//       逐行归属请以 patches/ 目录下的原始补丁为准。

// ---- mindspore/ccsrc/pipeline/jit/pi/graph_capture/graph_analyzer.cc  L140-L159 ----
bool GraphAnalyzer::HandleCallableToGraph(AObject *f) {
  if (f == nullptr) {
    return false;
  }
  // don't pass unknown callable to graph
  bool is_known_func = f->GetType() == AObject::kTypeCell || f->GetType() == AObject::kTypePrimitive ||
                       f->GetType() == AObject::kTypeMetaFuncGraph || CheckJitConstexpr(f->GetPyObject());
  bool is_ms_support_func = f->TestMsFlag(kMsFlagSet);
  if (!is_known_func && !is_ms_support_func) {
    return false;
  }
  if (f->GetType() == AObject::kTypePrimitive) {
    PyTypeObject *tp = f->GetTypeObject();
    std::string name = (tp && tp->tp_name ? tp->tp_name : "");
    if (name == "Assign") {
      return false;
    }
  }
  return true;
}

// ---- mindspore/ccsrc/pipeline/jit/pi/graph_capture/graph_analyzer.cc  L215-L229 ----
bool GraphAnalyzer::HandleSideEffectNodeForCapture(AbstractNode *capture_node) {
  auto &replace_map = graph_->GetSideEffect()->GetReplaceMap();
  bool find =
    std::any_of(replace_map.begin(), replace_map.end(),
                [&capture_node](std::pair<ValueNode *, ValueNode *> item) { return capture_node == item.second; });
  if (find) {
    return true;
  }
  for (auto &side_effect_node : graph_->GetSideEffect()->GetSideEffectNodes()) {
    if (capture_node == side_effect_node && side_effect_node->GetOpcode() == CALL_FUNCTION) {
      return true;
    }
  }
  return false;
}

// ---- mindspore/ccsrc/pipeline/jit/pi/graph_capture/graph_analyzer.cc  L353-L390 ----
void GraphAnalyzer::Analyze() {
  const FrameStates &enter_frame = graph_->GetFrame(0);
  GetCaptureInfo().escaped_locals.insert(enter_frame.GetLocals().begin(), enter_frame.GetLocals().end());
  AnalyzeRecursive(graph_);
  if (!HasTensorOperation()) {
    CleanCapturedValue();
  }
  UseDefAnalyze();
  if (graph_->GetStopTraceBci() != -1) {
    graph_->GetSideEffect()->CleanSideEffects(graph_->GetStopTraceBci());
  }
  CollectInputs();

  need_interpret_ = true;

  if (graph_->GetStopTraceBci() != -1 || !GetCaptureInfo().ordered_escaped_locals.empty()) {
    return;
  }
  bool support_ret = graph_->GetRetVal()->GetVobj() && graph_->GetRetVal()->GetVobj()->IsMindSporeSupportedType();
  if (!support_ret) {
    return;
  }
  PyCodeObject *co = graph_->GetCodeObj();
  const auto &args = enter_frame.GetLocals();
  int argc = co->co_argcount + co->co_kwonlyargcount;
  // check all parameters is graph supported, but here not check variable arguments
  auto end = args.begin() + argc;
  auto iter = std::find_if(args.begin(), end, [](ValueNode *i) { return !ValidateGraphParameters(i); });
  if (iter == end) {
    need_interpret_ = false;
  }
  if (!graph_->GetSideEffect()->GetSideEffectNodes().empty()) {
    need_interpret_ = true;
  }
  if (!graph_->GetSideEffect()->GetGlobalList().empty()) {
    need_interpret_ = true;
  }
}
