#!/usr/bin/env bash
#
# Rename this template to a real project.
#
#   ./scripts/init.sh my-service [github-owner]
#
# Replaces the REPO_NAME / REPO_NAME_UPPER placeholders everywhere, renames the
# files and directories that carry the name, tidies go.mod, and removes itself.
set -euo pipefail

name="${1:-}"
owner="${2:-pushkar-anand}"

if [[ -z "$name" ]]; then
	echo "usage: $0 <project-name> [github-owner]" >&2
	exit 1
fi

if ! [[ "$name" =~ ^[a-z][a-z0-9-]*$ ]]; then
	echo "error: project name must be lowercase and start with a letter: [a-z][a-z0-9-]*" >&2
	exit 1
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

self="scripts/$(basename "${BASH_SOURCE[0]}")"

# The env prefix and the Go import path have different shapes: build-with-go
# reads REPO_NAME_UPPER_ from the environment, and hyphens are not legal there.
upper="$(printf '%s' "$name" | tr 'a-z-' 'A-Z_')"

files() {
	if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
		git -C "$root" ls-files
	else
		find . -type f -not -path './.git/*' | sed 's|^\./||'
	fi
}

echo "==> renaming REPO_NAME -> $name (owner: $owner, env prefix: ${upper}_)"

while IFS= read -r f; do
	[[ "$f" == "$self" ]] && continue # this script defines the placeholders
	[[ -f "$f" ]] || continue

	# REPO_NAME_UPPER first: it contains REPO_NAME as a prefix.
	sed -e "s|REPO_NAME_UPPER|${upper}|g" \
		-e "s|pushkar-anand/REPO_NAME|${owner}/${name}|g" \
		-e "s|REPO_NAME|${name}|g" \
		"$f" >"$f.tmp"

	if cmp -s "$f" "$f.tmp"; then
		rm -f "$f.tmp"
	else
		mv "$f.tmp" "$f"
		echo "    $f"
	fi
done < <(files)

mv cmd/REPO_NAME "cmd/$name"
mv REPO_NAME.example.yaml "$name.example.yaml"
echo "    cmd/$name/, $name.example.yaml"

echo "==> go mod tidy"
go mod tidy

# The new name sorts differently from the placeholder, so re-group the imports.
echo "==> gofmt"
gofmt -w .

rm -f "$self"
rmdir scripts 2>/dev/null || true

cat <<EOF

Done. Next:

  cp $name.example.yaml $name.yaml   # local config, gitignored
  make build && make test

Set the module owner with a second argument if it is not $owner.
EOF
