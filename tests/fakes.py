"""A scripted fake model: you list the replies it will give, in order. No network, no cost."""
import itertools, json
from types import SimpleNamespace as NS

_ids = itertools.count(1)


def text(s):  return ("text", s)        # the model replies with plain text
def tool(code): return ("tool", code)   # the model asks to run this Python code
def boom(msg="busy"): return ("error", msg)   # the API call fails


class ScriptedClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        if not self.replies:
            raise AssertionError("the script ran out of replies: the agent made more model calls than expected")
        kind, value = self.replies.pop(0)
        if kind == "error":
            raise RuntimeError(value)
        usage = NS(prompt_tokens=10, completion_tokens=5)
        if kind == "tool":
            call = NS(id=f"c{next(_ids)}", function=NS(name="run_python", arguments=json.dumps({"code": value})))
            msg = NS(content="", tool_calls=[call])
        else:
            msg = NS(content=value, tool_calls=None)
        return NS(choices=[NS(message=msg)], usage=usage)
