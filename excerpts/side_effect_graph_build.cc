// Excerpt from MindSpore (https://github.com/mindspore-ai/mindspore)
// Copyright Huawei Technologies Co., Ltd. Licensed under the Apache License, Version 2.0.
// Snapshot: commit a8005072ba60ddeafffa84ea3d6953f95db179d0 (2024-03-26, merge of PR !67269)
// Contents: STORE_SUBSCR side effects: rewrite an assignment to a list element as a functional BUILD_LIST and register it in SideEffect
// Note: these are the complete bodies of functions that contain my changes, included for context;
//       not every line in them was written by me. For line-level authorship, see the original
//       patches in the patches/ directory.

// ---- mindspore/ccsrc/pipeline/jit/pi/graph_capture/graph_build.cc  L894-L934 ----
bool GraphBuilder::DoSideEffect(const Instr &instr, const std::vector<ValueNode *> &p) {
  // only handle list dict ; after mark sideeffect, erase value node
  auto key = p[0];
  auto container = p[1];
  auto value = p[2];
  ValueNode *new_node;
  PyObject *container_object = container->GetVobj()->GetPyObject().ptr();
  if (container_object == nullptr || !PyList_Check(container_object)) {
    return false;
  }
  std::vector<ValueNode *> items;
  Py_ssize_t py_list_size = PyList_Size(container_object);
  for (Py_ssize_t i = 0; i < py_list_size; i++) {
    // BINARY_SUBSCR
    ValueNode *index_node = NewValueNode(AObject::Convert(py::int_(i)), LOAD_CONST, -1, {});
    PyObject *py_object_item = PyList_GET_ITEM(container_object, i);
    ValueNode *item_node = NewValueNode(AObject::Convert(py_object_item), BINARY_SUBSCR, 0, {container, index_node});
    items.push_back(item_node);
  }
  PyObject *key_object = key->GetVobj()->GetPyObject().ptr();
  Py_ssize_t index = PyLong_AsSsize_t(key_object);
  Py_ssize_t size = py_list_size;
  if (index < -size || index >= size) {
    return false;
  }
  index = index < 0 ? (size + index) : index;
  AObject *object_info = AObject::BuildOperations(CollectObjects(items), BUILD_LIST);
  new_node = NewValueNode(object_info, BUILD_LIST, items.size(), items);
  if (!ReplaceAll(container, new_node)) {
    return false;
  }
  for (auto item : items) {
    graph_->GetTracedNodes().push_back(item);
  }
  items[index] = value;
  NewValueNode(nullptr, instr, {value, container, key});
  graph_->GetSideEffect()->SetSideEffectNode(new_node);
  graph_->GetSideEffect()->SetReplaceMap(new_node, container);
  graph_->GetTracedNodes().push_back(new_node);
  return true;
}

// ---- mindspore/ccsrc/pipeline/jit/pi/graph_capture/graph_build.cc  L936-L975 ----
bool GraphBuilder::DoItemAccess(const Instr &instr) {
  int opcode = instr.op();
  switch (opcode) {
    case BINARY_SUBSCR: {
      DoGetItem(instr);
      break;
    }
    case STORE_SUBSCR: {
      // STORE_SUBSCR: k->index v->item m-> container
      auto k = pop();
      auto m = pop();
      auto v = pop();
      if (DoSetItem(m, k, v) == true) {
        return true;
      }
      if (m->GetVobj()->GetType() == AObject::kTypeList) {
        if (k->GetOpcode() == BUILD_SLICE) {
          return false;
        }
        DoSideEffect(instr, {k, m, v});
        break;
      } else {
        NewValueNode(nullptr, instr, {v, m, k});
        current_block_->SetTrackResult(Block::kHasAttrSideEffect);
        break;
      }
    }
    case DELETE_SUBSCR: {
      auto sub = pop();  // sub
      auto obj = pop();  // obj
      obj->del_subscr(sub);
      NewValueNode(nullptr, instr, {obj, sub});
      current_block_->SetTrackResult(Block::kHasAttrSideEffect);
      break;
    }
    default:
      return false;
  }
  return true;
}
