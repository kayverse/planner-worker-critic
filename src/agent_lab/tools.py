"""The worker's one tool: run Python code. Includes the validator (rules checked BEFORE running).
NOTE: code still runs on your own computer. A sandbox comes in a later phase."""
import os, re, subprocess, sys, tempfile
from dataclasses import dataclass

TOOL = {"type": "function", "function": {"name": "run_python",
        "description": "Run Python 3 code in the data folder. Print what you need. pandas is available. "
                       "Variables and imports from your earlier SUCCESSFUL calls are still available, "
                       "so you do not need to reload the data every time.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}

BANNED = [r"os\.system", r"subprocess", r"shutil", r"socket", r"requests", r"urllib", r"__import__",
          r"\beval\(", r"\bexec\(", r"os\.remove", r"rmdir", r"open\([^)]*['\"][wax]"]


@dataclass
class ToolResult:
    output: str
    ok: bool
    blocked: bool = False


def validate(code: str):
    """Return a reason string if the code must not run, else None."""
    if not code.strip():
        return "empty code"
    for p in BANNED:
        if re.search(p, code):
            return f"blocked pattern: {p}"
    return None


def run_code(code: str, cwd: str, history=None, timeout: int = 30) -> ToolResult:
    """Run code in a fresh Python process. `history` = earlier successful code, replayed silently so
    variables (like df) persist between calls. Failed code is never added to the history."""
    bad = validate(code)
    if bad:
        return ToolResult(f"BLOCKED by validator: {bad}", False, blocked=True)
    script = code
    if history:
        prior = "\n".join(history)
        script = ("import io as _io, contextlib as _cl\n"
                  f"with _cl.redirect_stdout(_io.StringIO()):\n    exec(compile({prior!r}, '<earlier>', 'exec'), globals())\n") + code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
    try:
        p = subprocess.run([sys.executable, f.name], cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return ToolResult("timeout", False)
    finally:
        try:
            os.remove(f.name)
        except OSError:
            pass
    out = (p.stdout + p.stderr)[-3000:]
    ok = p.returncode == 0
    if ok and history is not None:
        history.append(code)
    if not ok:
        out += "\n(Note: this call failed, so variables it created were NOT kept. Reload the data in your next call.)"
    return ToolResult(out or "(no output)", ok)
