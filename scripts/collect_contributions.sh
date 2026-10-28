#!/usr/bin/env bash
# Re-extract all my commits from the MindSpore GitHub mirror and export them as
# patches, de-duplicated by patch-id.
#
# Usage: bash scripts/collect_contributions.sh [output-dir]
#
# Uses a treeless clone (--filter=tree:0) that downloads only commit objects (~50 MB);
# git fetches the trees and blobs it needs on demand when a diff is shown.
set -euo pipefail

AUTHOR_EMAIL="wuyunpu1@huawei.com"        # author names: wu-yunpu / 吴蕴溥 / w30010434
REPO_URL="https://github.com/mindspore-ai/mindspore"
OUT="${1:-./collected}"
CLONE="$OUT/mindspore.git"

mkdir -p "$OUT/patches"
if [ ! -d "$CLONE" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --bare --filter=tree:0 "$REPO_URL" "$CLONE"
fi
git -C "$CLONE" config gc.auto 0

echo "== All commits (all branches)"
git -C "$CLONE" log --all --no-merges --author="$AUTHOR_EMAIL" \
  --format='%h  %ad  %an  %s' --date=short | tee "$OUT/commits.txt"

echo
echo "== PRs I opened (from merge-commit messages)"
# Merge messages name the PR source as "<author>/<branch>"; my upstream author names are 吴蕴溥 and wu-yunpu.
git -C "$CLONE" log --all --merges --format='%h  %ad  %s' --date=short \
  | grep -E '吴蕴溥|wu-yunpu' | tee "$OUT/prs.txt"

echo
echo "== De-duplicate by patch-id and export"
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
echo "$n distinct patches exported to $OUT/patches/"
