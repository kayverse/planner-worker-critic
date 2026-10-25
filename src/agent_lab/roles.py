"""The three LLM roles. Each is a plain function that takes a RunContext and reports events."""
import json, re
from .tools import TOOL, run_code

PLANNER_SYS = ("You are the PLANNER. Break the question into 2-5 short, ordered subtasks for a data analyst who can run Python "
               "on a CSV file. Do not solve anything. The LAST subtask must produce the final answer. "
               'Reply with ONLY a JSON list of strings, e.g. ["...", "..."].')
WORKER_SYS = ("You are the WORKER analyst. Data file: {data} (in the current folder). Use the run_python tool; never guess numbers. "
              "Check the data for problems (duplicates, mixed formats, odd spelling, missing values) before trusting it. "
              "If the answer cannot be found in the data, say NOT FOUND. "
              "When you finish, reply with one line: 'FINAL: <answer>' if this is the last step, otherwise 'RESULT: <what you found>'.")
CRITIC_SYS = ("You are the CRITIC. Given a question, the work so far and a proposed answer, judge if it is correct and well supported. "
              "Look for: unhandled duplicates, mixed date formats, case/space issues, missing values, wrong filters, unsupported guesses. "
              'Reply with ONLY JSON: {"verdict": "accept"|"fix"|"replan", "note": "one short sentence"}. '
              "'fix' = redo this step with your note; 'replan' = the plan itself is wrong.")
NUDGE = ("Reply now with one line only: 'FINAL: <answer>' (or 'RESULT: <what you found>' if this is not the last step), "
         "based on the outputs above.")


def _text(msg) -> str:
    return msg.content or ""


def _echo_call(call_id, c) -> dict:
    """Copy a tool call back into the conversation. Gemini 3.x needs its hidden 'thought_signature'
    (kept in extra fields) sent back, so use model_dump() when the object has it."""
    if hasattr(c, "model_dump"):
        d = c.model_dump(exclude_none=True)
        d["id"], d["type"] = call_id, "function"
        return d
    return {"id": call_id, "type": "function",
            "function": {"name": c.function.name, "arguments": c.function.arguments}}


def make_plan(ctx, question, feedback=""):
    """PLANNER: turn the question into 2-5 ordered subtasks. Falls back to one step if the reply is not a JSON list."""
    q = question + (f"\nPrevious plan failed. Critic says: {feedback}" if feedback else "")
    reply = ctx.llm.chat(PLANNER_SYS, [{"role": "user", "content": q}])   # API errors propagate (no silent fallback)
    try:
        plan = json.loads(re.search(r"\[.*\]", _text(reply), re.S).group(0))
        plan = [str(s) for s in plan][:5] or [question]
    except Exception:
        plan = [question]
    ctx.log.emit("plan_created", steps=plan)
    return plan


def work(ctx, step, question, data_file, prior, feedback=""):
    """WORKER: do one subtask with the run_python tool. Returns (text, status) where status is 'done' or 'stuck'."""
    s = ctx.settings
    prompt = f"Overall question: {question}\nResults so far: {prior or 'none'}\nYour step: {step}"
    if feedback:
        prompt += f"\nCritic feedback to fix: {feedback}"
    ctx.log.emit("step_started", step=step)
    system = WORKER_SYS.format(data=data_file)
    msgs, bad_in_row, nudged = [{"role": "user", "content": prompt}], 0, False
    for _ in range(s.max_turns):
        msg = ctx.llm.chat(system, msgs, tools=[TOOL])
        calls = list(getattr(msg, "tool_calls", None) or [])
        if not calls:
            text = _text(msg)
            if not text.strip() and not nudged:        # model went silent: ask once for the answer line
                nudged = True
                ctx.log.emit("nudge", reason="empty reply")
                msgs.append({"role": "user", "content": NUDGE})
                continue
            ctx.log.emit("worker_reply", text=text)
            return text, "done"
        ids = [c.id or f"call_{i}" for i, c in enumerate(calls)]
        msgs.append({"role": "assistant", "content": msg.content or "",
                     "tool_calls": [_echo_call(i, c) for i, c in zip(ids, calls)]})
        for i, c in zip(ids, calls):
            try:
                code = json.loads(c.function.arguments).get("code", "")
            except Exception:
                code = ""
            res = run_code(code, ctx.base_dir, ctx.history)
            ctx.meter.steps += 1
            if res.blocked:
                ctx.meter.blocks += 1
            ctx.log.emit("tool_call", code=code[:500], ok=res.ok, blocked=res.blocked, output=res.output[:500])
            bad_in_row = 0 if res.ok else bad_in_row + 1
            msgs.append({"role": "tool", "tool_call_id": i, "content": res.output})
        if bad_in_row >= s.stuck_after:
            ctx.log.emit("worker_stuck", reason=f"{bad_in_row} failed code runs in a row")
            return "", "stuck"
    ctx.log.emit("out_of_turns")                         # out of tool turns: make it state its best answer
    msgs.append({"role": "user", "content": "You are out of tool turns. " + NUDGE})
    text = _text(ctx.llm.chat(system, msgs))
    if text.strip():
        ctx.log.emit("worker_reply", text=text)
        return text, "done"
    return "", "stuck"


def critic(ctx, question, plan, results, reason):
    """CRITIC: judge the work. Returns (verdict, note) with verdict 'accept', 'fix' or 'replan'."""
    ctx.meter.critic_calls += 1
    msg = f"Question: {question}\nPlan: {plan}\nWork/answer so far: {results}\nWhy you were called: {reason}"
    reply = ctx.llm.chat(CRITIC_SYS, [{"role": "user", "content": msg}], model=ctx.settings.critic)
    try:
        v = json.loads(re.search(r"\{.*\}", _text(reply), re.S).group(0))
        verdict, note = v.get("verdict", "accept"), v.get("note", "")
    except Exception:
        verdict, note = "accept", ""
    ctx.log.emit("critic_verdict", verdict=verdict, note=note, reason=reason)
    return verdict, note
