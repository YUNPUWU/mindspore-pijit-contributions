// Excerpt from MindSpore (https://github.com/mindspore-ai/mindspore)
// Copyright Huawei Technologies Co., Ltd. Licensed under the Apache License, Version 2.0.
// Snapshot: commit a8005072ba60ddeafffa84ea3d6953f95db179d0 (2024-03-26, merge of PR !67269)
// 内容: AbstractTuple 的切片/乘法类型推导
// 说明: 这里是包含我改动的完整函数体，便于阅读上下文；函数中并非每一行都由我编写，
//       逐行归属请以 patches/ 目录下的原始补丁为准。

// ---- mindspore/ccsrc/pipeline/jit/pi/graph_capture/abstract_object.cc  L846-L889 ----
AObject *AbstractTuple::Binary(AObject *o, int op) {
  // generic binary
  PyObject *r_obj = o ? o->GetPyObject().ptr() : nullptr;
  if (op == IS_OP) {
    bool cnst = const_object_type_map.find(r_obj) != const_object_type_map.end();
    return cnst ? Convert(Py_False) : MakeAObject(kTypeBool);
  }
  if (op == CONTAINS_OP) {
    return infer_binary_func[CONTAINS_OP](this, o);
  }
  // tuple binary
  if (o == nullptr || this->GetType() != o->GetType()) {
    if (this->GetType() == kTypeList && op == BINARY_MULTIPLY && (o != nullptr) && o->GetType() == kTypeInt) {
      AbstractTuple *ret = static_cast<AbstractTuple *>(MakeAObject(this->GetType()));
      std::vector<AObject *> temp;
      int res = PyLong_AsLong(o->GetPyObject().ptr());
      for (int i = 0; i < res; i++) {
        std::copy(items_.begin(), items_.end(), std::back_inserter(temp));
      }
      ret->Update(std::move(temp));
      return ret;
    }
    return MakeAObject(kTypeAnyValue);
  }
  AbstractTuple *r_list = static_cast<AbstractTuple *>(o);
  if (!this->IsElementValid() || !r_list->IsElementValid()) {
    return MakeAObject(kTypeAnyValue);
  }
  if (op == BINARY_ADD || (this->GetType() == kTypeTuple && op == INPLACE_ADD)) {
    AbstractTuple *ret = static_cast<AbstractTuple *>(MakeAObject(this->GetType()));
    std::vector<AObject *> temp;
    std::copy(items_.begin(), items_.end(), std::back_inserter(temp));
    std::copy(r_list->items_.begin(), r_list->items_.end(), std::back_inserter(temp));
    ret->Update(std::move(temp));
    return ret;
  }
  if (op == INPLACE_ADD) {
    std::copy(r_list->items_.begin(), r_list->items_.end(), std::back_inserter(this->items_));
    MarkModify();
    return this;
  }
  // binary mul, inplace mul
  return MakeAObject(kTypeAnyValue);
}

// ---- mindspore/ccsrc/pipeline/jit/pi/graph_capture/abstract_object.cc  L1092-L1134 ----
AObject *AbstractTuple::GetItem(AObject *k) {
  if (k == nullptr || k->GetType() == AObject::kTypeAnyValue || k->GetPyObject().ptr() == nullptr) {
    // user defined index or unknown key
    return this->AbstractSequence::GetItem(k);
  }
  if (!IsElementValid()) {
    return AObject::MakeAObject(element_type_);
  }
  if (k->GetType() == AObject::kTypeSlice) {
    if (this->GetPyObject().ptr() != nullptr) {
      return this->AbstractSequence::GetItem(k);
    }
    AObject *resultTuple = AObject::MakeAObject(this->type_);
    PyObject *slicePyObject = k->GetPyObject().ptr();
    Py_ssize_t start, stop, step;
    if (PySlice_Unpack(slicePyObject, &start, &stop, &step) < 0) {
      return AObject::MakeAObject(kTypeAnyValue);
    }
    if (start >= stop) {
      return resultTuple;
    }
    Py_ssize_t sliceLength = PySlice_AdjustIndices(this->items().size(), &start, &stop, step);
    AbstractTuple *resultTuplePtr = static_cast<AbstractTuple *>(resultTuple);
    if (start == 0 && step == 1 && sliceLength == this->size()) {
      return this;
    }
    if (step > 1) {
      int cursor = 0;
      std::vector<AObject *> itemsVector;
      for (cursor = 0; cursor < stop; cursor += step) {
        itemsVector.push_back(this->items()[cursor]);
      }
      resultTuplePtr->Update(itemsVector);
      return resultTuplePtr;
    }
    return AObject::MakeAObject(kTypeAnyValue);
  }
  Py_ssize_t index = GetTupleIndex(k, this->size());
  if (index == -1) {
    return AObject::MakeAObject(kTypeAnyValue);
  }
  return items_[index];
}
