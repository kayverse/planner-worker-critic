"""The control loop: planner -> worker (-> critic) with budgets. Returns the answer plus every event."""
import re, time
from dataclasses import dataclass
from .context import RunContext
from .events import EventLog
from .llm import LLM
from .meter import Meter
from .roles import critic, make_plan, work


@dataclass
class RunResult:
    answer: str
    meter: Meter
    events: list          # list of dicts, one per event


class Agent:
    def __init__(self, client, settings, base_dir=".", sleep=time.sleep):
        self.client, self.settings, self.base_dir, self.sleep = client, settings, base_dir, sleep

    def run(self, task, use_planner=True, use_critic=True, on_event=None) -> RunResult:
        s = self.settings
        meter, log = Meter(), EventLog(on_event)
        ctx = RunContext(LLM(self.client, s, meter, log, self.sleep), s, meter, log, self.base_dir)
        q, data_file = task["question"], task["data_file"]
        log.emit("run_started", question=q, planner=use_planner, critic=use_critic)

        plan = make_plan(ctx, q) if use_planner else [q]
        results, i, retries, feedback = [], 0, 0, ""
        while i < len(plan):
            text, status = work(ctx, plan[i], q, data_file, " | ".join(results), feedback)
            feedback = ""
            last = i == len(plan) - 1
            if (use_critic and meter.critic_calls < s.max_critic and retries < s.max_retries
                    and (status == "stuck" or last)):
                verdict, note = critic(ctx, q, plan, results + [text],
                                       "worker stuck" if status == "stuck" else "final answer review")
                if verdict == "fix":
                    retries += 1
                    feedback = note
                    continue
                if verdict == "replan" and use_planner:
                    retries += 1
                    plan = make_plan(ctx, q, note)
                    results, i = [], 0
                    continue
            results.append(text or "(step failed)")
            i += 1

        for r in reversed(results):                 # the answer is the last "FINAL: ..." line
            hit = re.findall(r"FINAL:\s*(.+)", r)
            if hit:
                answer = hit[-1].strip()
                break
        else:
            answer = results[-1] if results else ""
        log.emit("run_finished", answer=answer, steps=meter.steps, tokens=meter.tokens, critic_calls=meter.critic_calls)
        return RunResult(answer, meter, [e.to_dict() for e in log.events])
