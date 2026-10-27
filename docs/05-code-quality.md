# 05 graph_capture 静态检查清理

| 提交 | 日期 | PR | 补丁 |
| --- | --- | --- | --- |
| `fb8baeb` clean graph capture code check（主线） | 2024-03-25 | !67269 | [0011](../patches/05-code-quality/0011-clean-graph-capture-code-check.patch) |
| `e6ddc9d` 同上（r2.3 分支版本） | 2024-03-25 | !67140 | [variants](../patches/variants/r2.3-clean-graph-capture-code-check.patch) |
| `3870bf6` 同上（补一行格式） | 2024-03-25 | 经 !70706 同步进主线 | [0012](../patches/05-code-quality/0012-clean-graph-capture-code-check-followup.patch) |

7 个文件，+54 / −67 行，不改变功能，目的是让 `graph_capture/` 目录通过 MindSpore 合入流程中的静态检查（codecheck）。

## 处理的告警类型

| 类型 | 例子 | 处理方式 |
| --- | --- | --- |
| 有符号数参与位运算 | `tp->tp_flags & fast_type_mask`、`oparg & 0x08`、`track_result_ & (1 << kTrackBreak)` | 显式转换为 `unsigned` 后再做位运算 |
| 隐式窄化（`size_t` 转 `int`） | `int block_count = bb.size();`、`int nlocals = code_gen.GetLocalsMap().size();` | 改为 `static_cast<int>(...)` |
| 保留标识符 | 自定义宏 `_Py_MAKECODEUNIT` | 改名为 `PY_MAKECODEUNIT`。下划线加大写字母开头的名字是 C/C++ 保留给实现的；这个名字也和 CPython 头文件里的同名宏冲突，原代码只能靠 `#ifndef` 回避 |
| switch 分支缺少返回 | `IsRepeatWithoutSideEffect` 中 `case` 内部只有条件 return | 补上 `return false;`，避免隐式贯穿到 `default` |
| 类型不一致 | `code_flag`、`co_flags` 在 `int` 和 `unsigned` 之间混用 | `SetCodeFlags`、`MakeCapturedCode` 等接口统一用 `unsigned` |
| 遗留注释 | `BreakAtBlock` 中一段 `TODO` 注释块 | 删除 |

`cfg.h` 中 `Block` 的标志位判断是一个典型例子：

```cpp
// 修改前
bool IsTrackBreak() const { return track_result_ & (1 << kTrackBreak); }
// 修改后
bool IsTrackBreak() const { return track_result_ & (1 << (unsigned)kTrackBreak); }
```

## 回头看：机械地加类型转换的风险

这类清理很容易变成“哪里报警就在哪里加一个 cast”。回看这个补丁，有一处转换改变了语义：

```cpp
// abstract_object.cc, BUILD_CONST_KEY_MAP 分支
for (Py_ssize_t i = inputs.size() - 2; !err && i >= 0; --i)              // 修改前
for (Py_ssize_t i = (unsigned)(inputs.size() - 2); !err && i >= 0; --i)  // 修改后
```

当 `inputs.size() < 2` 时，修改前 `size_t` 下溢后再转成 `Py_ssize_t` 得到 -1，循环不执行；修改后先截断成 32 位无符号数 `0xFFFFFFFF`，再转成 64 位的 `Py_ssize_t`，得到一个很大的正数，循环会越界访问。CPython 正常不会生成没有键的 `BUILD_CONST_KEY_MAP`，所以实际很难触发，但更稳妥的写法是：

```cpp
for (Py_ssize_t i = static_cast<Py_ssize_t>(inputs.size()) - 2; !err && i >= 0; --i)
```

结论是：消除符号告警时，应该先把操作数转换成目标类型再做运算，而不是对运算结果做转换。
