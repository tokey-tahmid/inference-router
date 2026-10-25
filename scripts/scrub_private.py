"""Replace site-private identifiers in committed files with neutral placeholders.

    python scripts/scrub_private.py results/raw/*.json      # files, .gz handled by content
    python scripts/scrub_private.py --check results/raw/*   # report only, change nothing

Two git-ignored files drive it, so the identifiers themselves are never published:

* ``.private-patterns``: one regular expression per line, the same list the
  pre-commit check uses. Anything that still matches after the replacements
  below becomes ``<redacted>``.
* ``.private-replacements`` (optional): ``regex<TAB>placeholder`` per line,
  applied first and in order, so a storage path becomes ``${LIB_DATA}`` and a
  hostname keeps its node number rather than turning into an opaque marker.

Placeholders contain no quotes or backslashes, so JSON stays valid, and the
script checks that every ``.json`` file still parses before writing it. Only
identifying strings change: no measured value is touched. Prints the number of
replacements per changed file, so an unexpected match in data would show up.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rules() -> tuple[list[tuple[re.Pattern[str], str]], re.Pattern[str]]:
    pats_file = ROOT / ".private-patterns"
    pats = [p for p in pats_file.read_text().splitlines() if p.strip()] if pats_file.exists() else []
    if not pats:
        raise SystemExit("scrub_private: .private-patterns missing or empty; refusing to guess")
    rules = []
    repl_file = ROOT / ".private-replacements"
    if repl_file.exists():
        for line in repl_file.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                rx, _, placeholder = line.partition("\t")
                rules.append((re.compile(rx, re.IGNORECASE), placeholder))
    catch_all = re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
    return rules, catch_all


def scrub(text: str, rules, catch_all) -> tuple[str, int]:
    total = 0
    for rx, placeholder in rules:
        text, n = rx.subn(lambda m, p=placeholder: m.expand(p) if "\\" in p else p, text)
        total += n
    text, n = catch_all.subn("<redacted>", text)
    return text, total + n


def main() -> int:
    args = sys.argv[1:]
    check = "--check" in args
    rules, catch_all = _rules()
    changed = 0
    for name in (a for a in args if a != "--check"):
        f = Path(name)
        if not f.is_file() or f.suffix in (".png", ".jpg", ".pdf"):
            continue
        gz = f.suffix == ".gz"
        try:
            text = gzip.open(f, "rt").read() if gz else f.read_text()
        except UnicodeDecodeError:
            print(f"skip (binary): {f}")
            continue
        new, n = scrub(text, rules, catch_all)
        if not n:
            continue
        changed += 1
        print(f"{n:6d} replaced  {f}")
        if check:
            continue
        if f.suffix == ".json":
            json.loads(new)  # refuse to write a file the analysis could no longer read
        mode = f.stat().st_mode
        os.chmod(f, mode | stat.S_IWUSR)  # raw artifacts are written read-only
        if gz:
            with gzip.open(f, "wt", compresslevel=9) as out:
                out.write(new)
        else:
            f.write_text(new)
        os.chmod(f, mode)
    print(f"{changed} file(s) {'would change' if check else 'changed'}")
    return 1 if (check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
