#!/bin/bash
# Refuse to commit site-private identifiers into this (publishable) repository.
# Installed as .git/hooks/pre-commit by scripts/install_hooks.sh; also runnable by hand:
#   scripts/check_public.sh            # staged content
#   scripts/check_public.sh --all      # every tracked file, as it is on disk
# The pattern list lives in .private-patterns (git-ignored) so the patterns
# themselves never get published. scripts/scrub_private.py fixes what this finds.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
PATTERNS=.private-patterns
[[ -s $PATTERNS ]] || { echo "check_public: $PATTERNS missing; refusing to guess" >&2; exit 1; }
# Perl-style regexes where git supports them: POSIX extended regexes on some
# platforms silently ignore \b, which would let hostnames through unnoticed.
RX=-P
git grep -P -q -e x -- .no-such-path 2>/dev/null; [[ $? -eq 1 ]] || RX=-E
if [[ ${1:-} == --all ]]; then
    files() { git ls-files -z; }
    grep_text() { xargs -0 git grep -I -n -i "$RX" -f "$PATTERNS" -- ; }
    show() { cat -- "$1"; }
else
    files() { git diff --cached --name-only --diff-filter=ACM -z; }
    grep_text() { xargs -0 git grep --cached -I -n -i "$RX" -f "$PATTERNS" -- ; }
    show() { git show ":$1"; }
fi
# Text files go through git grep, which is fast over thousands of result files.
# Compressed files (packed logs) are checked by content.
hits=$( {
    files | grep -z -v '\.gz$' | grep_text
    files | grep -z '\.gz$' | while IFS= read -r -d '' f; do
        show "$f" | gzip -dc | grep -n -i -E -f "$PATTERNS" | sed "s|^|$f:|"
    done
} 2>/dev/null | cut -c1-240 )
if [[ -n $hits ]]; then
    echo "check_public: private identifiers found, commit blocked:" >&2
    echo "$hits" | head -50 | sed 's/^/  /' >&2
    echo "  ($(echo "$hits" | wc -l | tr -d ' ') line(s) in total)" >&2
    exit 1
fi
exit 0
