"""Pick the next phase from specs/PHASES.md. Prints RUN/STOP/DONE; exit 0=RUN, 1=STOP, 2=DONE."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROW = re.compile(r"^\|\s*(\d+)\s*\|(.+?)\|\s*(\S+)[^|]*\|\s*(auto|corpus|live)\s*\|\s*\[[^\]]*\]\(([^)]+)\)\s*\|")


def rows(text):
    return [(m[1], m[2].strip(), m[3], m[4], m[5]) for m in (ROW.match(l) for l in text.splitlines()) if m]


def pick(root=ROOT):
    """-> (code, message)."""
    if (root / "specs" / "BLOCKED.md").exists():
        return 1, "STOP blocked: specs/BLOCKED.md exists"
    todo = [r for r in rows((root / "specs" / "PHASES.md").read_text()) if r[2] != "✅"]
    if not todo:
        return 2, "DONE all phases complete"
    num, name, status, gate, path = todo[0]
    if status == "⛔":
        return 1, f"STOP phase {num} is blocked (see its file)"
    if gate == "live":
        return 1, f"STOP user step: phase {num} ({name}) needs the live game; see specs/phases/{Path(path).name}"
    if gate == "corpus" and not (root / "data" / "corpus" / "READY").exists():
        return 1, f"STOP phase {num} needs data/corpus/READY (finish phase 11)"
    return 0, f"RUN {num} specs/{path}"


if __name__ == "__main__":
    code, msg = pick()
    print(msg)
    sys.exit(code)
