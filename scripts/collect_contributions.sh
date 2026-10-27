#!/usr/bin/env bash
# 从 MindSpore 的 GitHub 镜像重新提取我的全部提交，并按 patch-id 去重导出补丁。
#
# 用法: bash scripts/collect_contributions.sh [输出目录]
#
# 使用无树克隆（--filter=tree:0），只下载提交对象，约 50 MB；
# 查看具体 diff 时 git 会按需拉取对应的 tree 和 blob。
set -euo pipefail

AUTHOR_EMAIL="wuyunpu1@huawei.com"        # 署名: wu-yunpu / 吴蕴溥 / w30010434
REPO_URL="https://github.com/mindspore-ai/mindspore"
OUT="${1:-./collected}"
CLONE="$OUT/mindspore.git"

mkdir -p "$OUT/patches"
if [ ! -d "$CLONE" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --bare --filter=tree:0 "$REPO_URL" "$CLONE"
fi
git -C "$CLONE" config gc.auto 0

echo "== 全部提交（所有分支）"
git -C "$CLONE" log --all --no-merges --author="$AUTHOR_EMAIL" \
  --format='%h  %ad  %an  %s' --date=short | tee "$OUT/commits.txt"

echo
echo "== 我发起的 PR（来自合并提交说明）"
git -C "$CLONE" log --all --merges --format='%h  %ad  %s' --date=short \
  | grep -E '吴蕴溥|wu-yunpu' | tee "$OUT/prs.txt"

echo
echo "== 按 patch-id 去重并导出"
declare -A seen=()
n=0
for c in $(git -C "$CLONE" log --all --no-merges --author="$AUTHOR_EMAIL" --reverse --format='%H'); do
  pid=$(git -C "$CLONE" show "$c" | git patch-id --stable | cut -d' ' -f1)
  if [ -z "${seen[$pid]:-}" ]; then
    seen[$pid]=1
    n=$((n + 1))
    subj=$(git -C "$CLONE" log -1 --format='%s' "$c" | tr -c 'A-Za-z0-9._-' '-' | sed 's/-\+/-/g; s/-$//')
    git -C "$CLONE" format-patch -1 --stdout --no-signature "$c" > "$OUT/patches/$(printf '%04d' $n)-${subj}.patch"
    echo "$(printf '%04d' $n)  ${c:0:9}  $(git -C "$CLONE" log -1 --format='%s' "$c")"
  fi
done
echo
echo "共 $n 个内容不同的补丁，已导出到 $OUT/patches/"
