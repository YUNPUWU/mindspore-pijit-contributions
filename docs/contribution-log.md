# Full Commit Log

Source: the GitHub mirror [mindspore-ai/mindspore](https://github.com/mindspore-ai/mindspore), filtered by author email `wuyunpu1@huawei.com` (author names `wu-yunpu`, `吴蕴溥`, `w30010434`). One merge commit that merged the upstream branch (`9a0ee49`) is excluded.

MindSpore is developed on Gitee, so PR numbers are Gitee numbers (`!xxxxx`). A change was usually merged into a feature branch first and then brought into mainline by a teammate's sync PR, so the same patch appears as different commits on several branches. After de-duplicating with `git patch-id` there are 12 mainline patches and 3 branch versions with different content.

| Date | Commit | Author name | Title | Branch / how it was merged | PR I opened | Diff | Patch in this repo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-11-25 | [`63863e71c`](https://github.com/mindspore-ai/mindspore/commit/63863e71c8f6205c3e9ee5ce45762f504cc27aca) | w30010434 | support tuple type derive | mainline (merged via !65105; reverted earlier on the feature branch by !62564) | — | 1 file +25 / −4 | 0001 |
| 2023-11-25 | [`41e7e2046`](https://github.com/mindspore-ai/mindspore/commit/41e7e2046897c44b218ddf645037249223f4dca7) | w30010434 | support tuple type derive | mstorch-infer-one-stage branch | [!62405](https://gitee.com/mindspore/mindspore/pulls/62405) | 1 file +25 / −4 | — |
| 2023-12-12 | [`c9f43f991`](https://github.com/mindspore-ai/mindspore/commit/c9f43f991dd823008e75c7580d3f9d02786e83c8) | wu-yunpu | support tuple slice type inference | mainline (merged via !65105) | — | 1 file +39 / −0 | 0002 |
| 2023-12-12 | [`f5ffb71c2`](https://github.com/mindspore-ai/mindspore/commit/f5ffb71c2c9bad16488e39397adc698909791887) | wu-yunpu | support tuple slice type inference | mstorch-infer-one-stage branch | [!63085](https://gitee.com/mindspore/mindspore/pulls/63085) | 1 file +39 / −0 | — |
| 2023-12-30 | [`8f94ed7cf`](https://github.com/mindspore-ai/mindspore/commit/8f94ed7cf4935b1fe2f5afde2b5ff44e9313f8f8) | wu-yunpu | add loop trace switch and store subscr default crack map | mainline (merged via !65105) | — | 7 files +31 / −29 | 0003 |
| 2023-12-30 | [`4244b989a`](https://github.com/mindspore-ai/mindspore/commit/4244b989af8a2694f2316c082da72a553078f575) | wu-yunpu | add loop trace switch and store subscr default crack map | mstorch-infer-one-stage branch | [!63765](https://gitee.com/mindspore/mindspore/pulls/63765) | 7 files +31 / −29 | — |
| 2024-01-05 | [`b053d50ef`](https://github.com/mindspore-ai/mindspore/commit/b053d50efe0efde5f57b4416726d12471299a6ab) | wu-yunpu | fix LIST EXTEND failed and kwonlyargs bug | mainline (merged via !65105) | — | 2 files +16 / −1 | 0004 |
| 2024-01-05 | [`d02e2f130`](https://github.com/mindspore-ai/mindspore/commit/d02e2f1303a39160c8ae16ddf25ddcd69987bb08) | wu-yunpu | fix LIST EXTEND failed and kwonlyargs bug | mstorch-infer-one-stage branch | [!63896](https://gitee.com/mindspore/mindspore/pulls/63896) | 2 files +16 / −1 | — |
| 2024-01-06 | [`584a5191c`](https://github.com/mindspore-ai/mindspore/commit/584a5191caf4a3b11ff8491717369fd66cf3aacb) | wu-yunpu | fix listextend bugs | mainline (merged via !65105) | — | 1 file +2 / −2 | 0005 |
| 2024-01-06 | [`df23faf98`](https://github.com/mindspore-ai/mindspore/commit/df23faf987dd558d921556598199cea9365b9029) | wu-yunpu | fix listextend bugs | mstorch-infer-one-stage branch | [!63919](https://gitee.com/mindspore/mindspore/pulls/63919) | 1 file +2 / −2 | — |
| 2024-02-02 | [`38440d6d9`](https://github.com/mindspore-ai/mindspore/commit/38440d6d92a30bfd1b7e0e395ce23763394c4f19) | wu-yunpu | elimate store deref | mainline (merged via !65105) | — | 1 file +14 / −1 | 0006 |
| 2024-02-02 | [`e2cc083b5`](https://github.com/mindspore-ai/mindspore/commit/e2cc083b5ca873d45f117288f37a1c8a84ee8743) | wu-yunpu | elimate store deref | mstorch-infer-one-stage branch | [!65004](https://gitee.com/mindspore/mindspore/pulls/65004) | 1 file +14 / −1 | — |
| 2024-02-02 | [`5e2f080a0`](https://github.com/mindspore-ai/mindspore/commit/5e2f080a07750126244f3323e166677cd88790fb) | wu-yunpu | support side effect | mstorch-infer branch | [!65841](https://gitee.com/mindspore/mindspore/pulls/65841) | 11 files +282 / −8 | variants |
| 2024-02-02 | [`c54f56ebb`](https://github.com/mindspore-ai/mindspore/commit/c54f56ebba214c6e2210083a0d2cb0df71c7b099) | wu-yunpu | support side effect | mainline (merged via !65923) | — | 12 files +283 / −251 | 0007 |
| 2024-03-04 | [`819c3e79f`](https://github.com/mindspore-ai/mindspore/commit/819c3e79f279d60a12352f892c7ede6bb870d3cd) | wu-yunpu | delete fallback st | mstorch-infer branch | [!65841](https://gitee.com/mindspore/mindspore/pulls/65841) | 3 files +9 / −247 | variants |
| 2024-03-11 | [`155e73f50`](https://github.com/mindspore-ai/mindspore/commit/155e73f50898a300a1698aa6c470bf9af6851b92) | 吴蕴溥 | add side effect ut | mainline | [!66290](https://gitee.com/mindspore/mindspore/pulls/66290) | 1 file +154 / −0 | 0008 |
| 2024-03-12 | [`a08068927`](https://github.com/mindspore-ai/mindspore/commit/a08068927d5e939aa49fa0b24b2a0c50ba18a235) | wu-yunpu | fix side effect bug | mainline | [!66488](https://gitee.com/mindspore/mindspore/pulls/66488) | 12 files +246 / −201 | 0009 |
| 2024-03-22 | [`e24592a22`](https://github.com/mindspore-ai/mindspore/commit/e24592a2211ee847f004f0b8b3c89db55bd6fac7) | 吴蕴溥 | fix py3.7 side effect case bug | mainline | [!66980](https://gitee.com/mindspore/mindspore/pulls/66980) | 2 files +1 / −4 | 0010 |
| 2024-03-22 | [`9cfac6ed5`](https://github.com/mindspore-ai/mindspore/commit/9cfac6ed5b334a1dc21813d5020a589e12b088d5) | 吴蕴溥 | fix side effect py3.7 case bug | r2.3 branch | [!66983](https://gitee.com/mindspore/mindspore/pulls/66983) | 2 files +1 / −4 | — |
| 2024-03-25 | [`fb8baeb1f`](https://github.com/mindspore-ai/mindspore/commit/fb8baeb1fc7746ba792dceaa4dedf47d9758e123) | wu-yunpu | clean graph capture code check | mainline | [!67269](https://gitee.com/mindspore/mindspore/pulls/67269) | 7 files +54 / −67 | 0011 |
| 2024-03-25 | [`e6ddc9da9`](https://github.com/mindspore-ai/mindspore/commit/e6ddc9da9468257203bc61dbb09598c9174b8824) | wu-yunpu | clean graph capture code check | r2.3 branch | [!67140](https://gitee.com/mindspore/mindspore/pulls/67140) | 7 files +54 / −65 | variants |
| 2024-03-25 | [`3870bf675`](https://github.com/mindspore-ai/mindspore/commit/3870bf675da68472e30db1b6174cae6ac4181ebc) | wu-yunpu | clean graph capture code check | mainline (merged via !70706) | — | 1 file +1 / −0 | 0012 |
| 2024-03-25 | [`e100e6515`](https://github.com/mindspore-ai/mindspore/commit/e100e6515854e654dd9c1df8f5541047d6573b4a) | wu-yunpu | clean graph capture code check | other feature branches | — | 1 file +1 / −0 | — |

## The 13 PRs I opened

| PR | Title (verbatim) |
| --- | --- |
| [!62405](https://gitee.com/mindspore/mindspore/pulls/62405) | support tuple derive |
| [!63085](https://gitee.com/mindspore/mindspore/pulls/63085) | support tuple type slice inference |
| [!63765](https://gitee.com/mindspore/mindspore/pulls/63765) | refactor codegen and support loop trace by chaiyouheng (also contains commits by my teammate chaiyouheng) |
| [!63896](https://gitee.com/mindspore/mindspore/pulls/63896) | fix LIST EXTEND failed and kwonlyargs bug |
| [!63919](https://gitee.com/mindspore/mindspore/pulls/63919) | fix extend list |
| [!65004](https://gitee.com/mindspore/mindspore/pulls/65004) | eslimate store deref |
| [!65841](https://gitee.com/mindspore/mindspore/pulls/65841) | yellow support store subscr side effect |
| [!66290](https://gitee.com/mindspore/mindspore/pulls/66290) | add side effect ut to r2.3 |
| [!66488](https://gitee.com/mindspore/mindspore/pulls/66488) | optimization side effect and fix bug |
| [!66980](https://gitee.com/mindspore/mindspore/pulls/66980) | fix py3.7 side effect bug q1 |
| [!66983](https://gitee.com/mindspore/mindspore/pulls/66983) | fix py2.3 side effect |
| [!67140](https://gitee.com/mindspore/mindspore/pulls/67140) | clear graph capture code check |
| [!67269](https://gitee.com/mindspore/mindspore/pulls/67269) | codecheck capture q1 |

Sync PRs opened by teammates that carried my commits: !65105 (hejianheng, merge pijit project), !65923 (hejianheng, update pijit code 3/4), !70706 (shaoshengqi, merge pijit one-stage features into master).

PR titles and merge information come from the merge-commit messages generated by `i-robot` / maintainers in the mirror. The Gitee links are built from the PR numbers and were not opened one by one to check.
