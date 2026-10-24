"""Planner + Worker + Critic agent for ANY OpenAI-compatible API (Gemini, Groq, OpenRouter, ...). Needs: pip install openai"""
import os, json, re, subprocess, sys, tempfile, time

MODEL = os.environ.get("MODEL", "gemini-2.5-flash")      # confirm the exact name in your provider's console
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", MODEL)     # a different model here reduces "critic agrees with worker"
PRICE_IN = float(os.environ.get("PRICE_IN", 0))          # $ per 1M input tokens (0 on a free tier -> use tokens/task instead)
PRICE_OUT = float(os.environ.get("PRICE_OUT", 0))
TEMPERATURE = os.environ.get("TEMPERATURE", "0")         # set to none if the model rejects it
DELAY = float(os.environ.get("DELAY", 0))                # seconds to wait after every call (helps with per-minute limits)
MAX_TURNS, MAX_CRITIC, MAX_RETRIES = 6, 3, 2             # budgets (per task)
DEBUG = os.environ.get("DEBUG")                          # set DEBUG=1 to print every code run and its output

TOOL = {"type": "function", "function": {"name": "run_python",
        "description": "Run Python 3 code in the data folder. Print what you need. pandas is available. Variables and imports from your earlier SUCCESSFUL calls are still available, so you do not need to reload the data every time.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}

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

BANNED = [r"os\.system", r"subprocess", r"shutil", r"socket", r"requests", r"urllib", r"__import__", r"\beval\(", r"\bexec\(",
          r"os\.remove", r"rmdir", r"open\([^)]*['\"][wax]"]

class Meter:
    def __init__(self): self.tin = self.tout = self.steps = self.critic_calls = self.blocks = 0
    @property
    def cost(self): return (self.tin * PRICE_IN + self.tout * PRICE_OUT) / 1e6

def validate(code):
    """Gate-1: cheap rule check BEFORE the worker's code runs."""
    if not code.strip(): return "empty code"
    for p in BANNED:
        if re.search(p, code): return f"blocked pattern: {p}"
    return None

def safe_run(code, m, cwd, history=None):
    bad = validate(code)
    if bad: m.blocks += 1; return f"BLOCKED by validator: {bad}", False
    script = code
    if history:   # replay earlier successful code silently so variables (e.g. df) persist between calls
        prior = "\n".join(history)
        script = ("import io as _io, contextlib as _cl\n"
                  f"with _cl.redirect_stdout(_io.StringIO()):\n    exec(compile({prior!r}, '<earlier>', 'exec'), globals())\n") + code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f: f.write(script)
    try:
        p = subprocess.run([sys.executable, f.name], cwd=cwd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
        out = (p.stdout + p.stderr)[-3000:]
        ok = p.returncode == 0
        if ok and history is not None: history.append(code)
        if not ok: out += "\n(Note: this call failed, so variables it created were NOT kept. Reload the data in your next call.)"
        return out or "(no output)", ok
    except subprocess.TimeoutExpired: return "timeout", False

def call(client, m, system, messages, model=None, tools=None):
    kw = dict(model=model or MODEL, messages=[{"role": "system", "content": system}] + messages, max_tokens=4000)
    if tools: kw["tools"] = tools
    if TEMPERATURE.lower() != "none": kw["temperature"] = float(TEMPERATURE)
    for attempt in range(6):                      # retry on rate limits / temporary errors
        try: r = client.chat.completions.create(**kw); break
        except Exception as e:
            if attempt == 5: raise
            print(f"  (API error, retrying: {str(e)[:120]})"); time.sleep(min(60, 5 * 2 ** attempt))
    u = r.usage
    m.tin += getattr(u, "prompt_tokens", 0) or 0; m.tout += getattr(u, "completion_tokens", 0) or 0
    if DELAY: time.sleep(DELAY)
    return r.choices[0].message

def text_of(msg): return msg.content or ""

def make_plan(client, m, question, feedback=""):
    q = question + (f"\nPrevious plan failed. Critic says: {feedback}" if feedback else "")
    try:
        plan = json.loads(re.search(r"\[.*\]", text_of(call(client, m, PLANNER_SYS, [{"role": "user", "content": q}])), re.S).group(0))
        return [str(s) for s in plan][:5] or [question]
    except Exception: return [question]

def work(client, m, step, question, data_file, cwd, prior, feedback="", history=None):
    prompt = f"Overall question: {question}\nResults so far: {prior or 'none'}\nYour step: {step}"
    if feedback: prompt += f"\nCritic feedback to fix: {feedback}"
    msgs, bad_in_row, nudged = [{"role": "user", "content": prompt}], 0, False
    NUDGE = "Reply now with one line only: 'FINAL: <answer>' (or 'RESULT: <what you found>' if this is not the last step), based on the outputs above."
    for _ in range(MAX_TURNS):
        msg = call(client, m, WORKER_SYS.format(data=data_file), msgs, tools=[TOOL])
        calls = list(getattr(msg, "tool_calls", None) or [])
        if not calls:
            if not text_of(msg).strip() and not nudged:      # model went silent: ask once for the answer line
                nudged = True; msgs.append({"role": "user", "content": NUDGE}); continue
            return text_of(msg), "done"
        ids = [c.id or f"call_{i}" for i, c in enumerate(calls)]
        def _echo(i, c):
            # Gemini 3.x needs its hidden "thought_signature" (kept in extra fields) sent back with each tool call
            if hasattr(c, "model_dump"):
                d = c.model_dump(exclude_none=True); d["id"] = i; d["type"] = "function"; return d
            return {"id": i, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
        msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [_echo(i, c) for i, c in zip(ids, calls)]})
        for i, c in zip(ids, calls):
            try: code = json.loads(c.function.arguments).get("code", "")
            except Exception: code = ""
            out, ok = safe_run(code, m, cwd, history); m.steps += 1
            if DEBUG: print(f"\n--- code (ok={ok}) ---\n{code[:500]}\n--- output ---\n{out[:500]}")
            bad_in_row = 0 if ok else bad_in_row + 1
            msgs.append({"role": "tool", "tool_call_id": i, "content": out})
        if bad_in_row >= 3: return "", "stuck"
    # out of tool turns: make the model state its answer from what it has
    msgs.append({"role": "user", "content": "You are out of tool turns. " + NUDGE})
    msg = call(client, m, WORKER_SYS.format(data=data_file), msgs)
    return (text_of(msg), "done") if text_of(msg).strip() else ("", "stuck")

def critic(client, m, question, plan, results, reason):
    m.critic_calls += 1
    msg = f"Question: {question}\nPlan: {plan}\nWork/answer so far: {results}\nWhy you were called: {reason}"
    try:
        v = json.loads(re.search(r"\{.*\}", text_of(call(client, m, CRITIC_SYS, [{"role": "user", "content": msg}], CRITIC_MODEL)), re.S).group(0))
        return v.get("verdict", "accept"), v.get("note", "")
    except Exception: return "accept", ""

def run_task(client, task, use_planner, use_critic, base_dir="."):
    m, q, df = Meter(), task["question"], task["data_file"]
    plan = make_plan(client, m, q) if use_planner else [q]
    results, i, retries, feedback, hist = [], 0, 0, "", []
    while i < len(plan):
        txt, status = work(client, m, plan[i], q, df, base_dir, " | ".join(results), feedback, hist); feedback = ""
        last = i == len(plan) - 1
        if use_critic and m.critic_calls < MAX_CRITIC and retries < MAX_RETRIES and (status == "stuck" or last):
            v, note = critic(client, m, q, plan, results + [txt], "worker stuck" if status == "stuck" else "final answer review")
            if v == "fix": retries += 1; feedback = note; continue
            if v == "replan" and use_planner:
                retries += 1; plan = make_plan(client, m, q, note); results, i = [], 0; continue
        results.append(txt or "(step failed)"); i += 1
    joined = " ".join(results)
    hit = re.findall(r"FINAL:\s*(.+)", joined)
    return (hit[-1].strip() if hit else results[-1] if results else ""), m
