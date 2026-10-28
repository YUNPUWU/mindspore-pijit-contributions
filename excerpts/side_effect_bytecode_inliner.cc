// Excerpt from MindSpore (https://github.com/mindspore-ai/mindspore)
// Copyright Huawei Technologies Co., Ltd. Licensed under the Apache License, Version 2.0.
// Snapshot: commit a8005072ba60ddeafffa84ea3d6953f95db179d0 (2024-03-26, merge of PR !67269)
// Contents: Inlining: merge sub-graph side effects into the top-level graph; remove redundant STORE_DEREF
// Note: these are the complete bodies of functions that contain my changes, included for context;
//       not every line in them was written by me. For line-level authorship, see the original
//       patches in the patches/ directory.

// ---- mindspore/ccsrc/pipeline/jit/pi/graph_capture/bytecode_inliner.cc  L172-L191 ----
void BytecodeInliner::CollectTracedNodes(Graph *graph) {
  for (ValueNode *n : graph->GetTracedNodes()) {
    if (n->GetType() != AbstractNode::Call) {
      traced_nodes_.push_back(n);
      continue;
    }
    CallNode *call_node = static_cast<CallNode *>(n);
    if (call_node->GetSubGraph() == nullptr || call_node->GetInlineReason() != InlineReason::kInline) {
      traced_nodes_.push_back(n);
      continue;
    }
    std::copy(call_node->GetParams().begin(), call_node->GetParams().end(), std::back_inserter(traced_nodes_));
    CollectTracedNodes(call_node->GetSubGraph());
  }
  if (graph != graph_) {
    if (graph->GetSideEffect() != nullptr) {
      graph_->GetSideEffect()->Merge(graph->GetSideEffect());
    }
  }
}

// ---- mindspore/ccsrc/pipeline/jit/pi/graph_capture/bytecode_inliner.cc  L459-L506 ----
void BytecodeInliner::EliminateClosureSideEffect() {
  PyCodeObject *co = graph_->GetCodeObj();
  int ncells = PyTuple_GET_SIZE(co->co_cellvars);
  int nfrees = PyTuple_GET_SIZE(co->co_freevars);
  if (ncells + nfrees == 0) {
    return;
  }
  std::set<InstrNode *> alive_closure_access;

  if (last_frame_ != nullptr) {
    auto iter = std::find_if(cfg_->instr_pool().begin(), cfg_->instr_pool().end(), [](const std::unique_ptr<Instr> &i) {
      return i->op() == LOAD_DEREF || (i->op() == MAKE_FUNCTION && ((signed)i->arg() & 0x08));
    });
    if (iter != cfg_->instr_pool().end()) {
      return;
    }
  }

  for (auto i : traced_nodes_) {
    if (i->GetOpcode() == MAKE_FUNCTION && (i->GetOparg() & 0x08)) {
      ValueNode *tuple = *(i->getInputs().end() - 3);
      for (auto c : tuple->getInputs()) {
        const auto &nodes = static_cast<CellVarNode *>(c)->GetCellOper();
        alive_closure_access.insert(nodes.begin(), nodes.end());
      }
    }
  }

  auto iter = std::remove_if(traced_nodes_.begin(), traced_nodes_.end(), [&alive_closure_access](ValueNode *i) {
    int op = i->GetOpcode();
    return (op == STORE_DEREF || op == DELETE_DEREF) && alive_closure_access.find(i) == alive_closure_access.end();
  });
  traced_nodes_.erase(iter, traced_nodes_.end());

  for (auto item = traced_nodes_.begin(); item != traced_nodes_.end();) {
    if ((*item)->GetOpcode() == STORE_DEREF) {
      if ((*item)->getInputs()[0]->GetOpcode() == LOAD_DEREF &&
          (*item)->getInputs()[0]->GetOparg() == (*item)->GetOparg() &&
          (*item)->getInputs()[0]->GetGraph() == (*item)->GetGraph()) {
        item = traced_nodes_.erase(item);
      } else {
        ++item;
      }
    } else {
      ++item;
    }
  }
}
