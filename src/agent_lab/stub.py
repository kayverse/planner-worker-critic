"""Fake model, ONLY to test the plumbing offline (python src/run.py --stub). Its answers mean nothing."""
from types import SimpleNamespace as NS

U = NS(prompt_tokens=100, completion_tokens=50)


def _r(content=None, tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))], usage=U)


class StubClient:
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, model, messages, max_tokens=0, tools=None, temperature=None):
        system = messages[0]["content"]
        if "PLANNER" in system:
            return _r('["load data", "answer"]')
        if "CRITIC" in system:
            return _r('{"verdict": "accept", "note": ""}')
        if not any(m["role"] == "tool" for m in messages):
            return _r("", [NS(id="t1", function=NS(name="run_python", arguments='{"code": "print(1)"}'))])
        return _r("FINAL: 0")
