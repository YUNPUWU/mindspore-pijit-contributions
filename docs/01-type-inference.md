# 01 Type Inference for Tuple Slicing and List Repetition

| Commit | Date | PR | Patch |
| --- | --- | --- | --- |
| `63863e7` support tuple type derive | 2023-11-25 | !62405 | [0001](../patches/01-type-inference/0001-support-tuple-type-derive.patch) |
| `c9f43f9` support tuple slice type inference | 2023-12-12 | !63085 | [0002](../patches/01-type-inference/0002-support-tuple-slice-type-inference.patch) |

Changed file: `graph_capture/abstract_object.cc` (`AbstractTuple::GetItem`, `AbstractTuple::Binary`). The full functions are in [excerpts/type_inference_abstract_object.cc](../excerpts/type_inference_abstract_object.cc).

## Background

For every bytecode instruction it processes, `GraphBuilder` computes an abstract object (`AObject`) for the result. Tuples and lists are both represented by `AbstractTuple`, which records the `AObject` of each element. As long as the element types are known, later operations such as `t[0] + 1` or passing an element to an operator can still be inferred and put into the graph.

Before this change, `AbstractTuple::GetItem` only supported integer indices:

```cpp
Py_ssize_t index = GetTupleIndex(k, this->size());
if (index == -1) {
  return AObject::MakeAObject(kTypeAnyValue);   // a slice ends up here and becomes "unknown"
}
return items_[index];
```

So the result of a slice such as `t[::2]` could only be inferred as `kTypeAnyValue`, and every node depending on it lost its type information.

## The first version and its problem (!62405)

The first version taught `GetItem` to handle `kTypeSlice` keys: it used `PySlice_Unpack` to get start / stop / step and built the resulting tuple. The problem was that it **replaced** the existing integer-index branch: a non-slice key returned an empty tuple instead of the element. This was a regression, and on 2023-11-29 the maintainers reverted it in !62564 (`Revert "support tuple type derive"`).

The lesson I took from the revert: when adding a branch to existing logic, leave the original path untouched, and make sure tests cover the original path.

## The second version: adding a branch incrementally (!63085)

The second version only enters the new logic when the key is a slice; not a single line of the integer-index path changed:

```cpp
if (k->GetType() == AObject::kTypeSlice) {
  if (this->GetPyObject().ptr() != nullptr) {
    return this->AbstractSequence::GetItem(k);       // real object available: let the generic code slice it
  }
  AObject *resultTuple = AObject::MakeAObject(this->type_);   // keep tuple / list type unchanged
  Py_ssize_t start, stop, step;
  if (PySlice_Unpack(slicePyObject, &start, &stop, &step) < 0) {
    return AObject::MakeAObject(kTypeAnyValue);
  }
  if (start >= stop) {
    return resultTuple;                               // empty slice
  }
  Py_ssize_t sliceLength = PySlice_AdjustIndices(this->items().size(), &start, &stop, step);
  if (start == 0 && step == 1 && sliceLength == this->size()) {
    return this;                                      // t[:], reuse as is
  }
  if (step > 1) { ... pick elements by step ... }
  return AObject::MakeAObject(kTypeAnyValue);         // anything else conservatively becomes unknown
}
// the original integer-index logic follows, unchanged
```

Design considerations:

- **Use the real value when there is one.** If the tuple has a corresponding Python object at trace time (for example, a constant tuple), the generic `AbstractSequence::GetItem` can slice it and get an exact result, so there is nothing to infer. The new logic is only for tuples whose element types are known but whose concrete value is not.
- **The result type follows the original container.** The result is created with `this->type_`, so slicing a list still gives a list and slicing a tuple gives a tuple (the first version hard-coded `kTypeTuple`).
- **CPython's `PySlice_AdjustIndices` normalizes the indices**, handling negative indices and out-of-range bounds with the same semantics as Python.
- **A full slice returns `this`**, so `t[:]` does not copy the abstract object.
- **When unsure, fall back.** Any case that is not implemented returns `kTypeAnyValue`: better to put less into the graph than to infer a wrong type.

### List repetition

The same PR also added inference for `list * int` to `AbstractTuple::Binary`:

```cpp
if (this->GetType() == kTypeList && op == BINARY_MULTIPLY && o != nullptr && o->GetType() == kTypeInt) {
  // [a, b] * 3  ->  [a, b, a, b, a, b], copying each element's abstract object
}
```

After the common initialization idiom `[x] * n`, the type of every element is still known.

## Example

```python
@jit(mode="PIJit")
def f(a, b, c, d):          # a, b, c, d are Tensors
    t = (a, b, c, d)
    u = t[::2]              # before: AnyValue; after: (Tensor, Tensor)
    return u[0] + u[1]      # after the change, this can still go into the graph
```

## Known limitations

- In the `step > 1` branch the cursor starts at 0 and ignores `start`, so `t[1::2]` picks the wrong elements. The correct loop is `for (cursor = start; cursor < stop; cursor += step)`, or simply counting with `sliceLength`.
- Partial slices with `step == 1` (such as `t[1:3]`) and negative steps are not implemented and fall back to an unknown type.
- `list * int` does not check whether the integer's value is known. When `o->GetPyObject()` is null, `PyLong_AsLong` returns -1 and leaves a Python exception set. It should check for null first and fall back to an unknown type if the value is unavailable.
