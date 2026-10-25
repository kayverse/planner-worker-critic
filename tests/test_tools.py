from agent_lab.tools import run_code, validate


def test_validate_blocks_dangerous_code():
    assert validate("import subprocess") is not None
    assert validate("open('x.txt', 'w')") is not None
    assert validate("   ") == "empty code"
    assert validate("print(1)") is None


def test_run_code_returns_output(tmp_path):
    r = run_code("print(1 + 1)", str(tmp_path))
    assert r.ok and r.output.strip() == "2"


def test_variables_persist_between_calls(tmp_path):
    h = []
    run_code("x = 41", str(tmp_path), h)
    r = run_code("print(x + 1)", str(tmp_path), h)
    assert r.output.strip() == "42"


def test_earlier_code_runs_silently(tmp_path):
    h = []
    run_code("print('loud')", str(tmp_path), h)
    assert run_code("print('quiet')", str(tmp_path), h).output.strip() == "quiet"


def test_failed_code_is_not_kept(tmp_path):
    h = []
    r = run_code("y = 1\nprint(undefined_name)", str(tmp_path), h)
    assert not r.ok and h == [] and "NOT kept" in r.output


def test_blocked_code_is_flagged(tmp_path):
    r = run_code("import shutil", str(tmp_path))
    assert r.blocked and not r.ok


def test_timeout(tmp_path):
    r = run_code("while True: pass", str(tmp_path), timeout=1)
    assert r.output == "timeout" and not r.ok
