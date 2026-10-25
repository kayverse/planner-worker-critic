"""Test runner. Usage (from the project folder):
    python src\run.py --stub                              offline check with a fake model
    python src\run.py --configs worker --limit 3          real run on 3 tasks
    python src\run.py --configs worker,full --verbose     print every event as it happens
Progress is saved in results/results.jsonl after every task; finished tasks are skipped on re-run."""
import argparse, collections, json, os
try:
    from dotenv import load_dotenv
    load_dotenv()                       # reads the .env file (your key lives there)
except ImportError:
    pass
from agent_lab import Agent, Settings
from agent_lab.check import check

CONFIGS = {"worker": (False, False), "planner": (True, False), "critic": (False, True), "full": (True, True)}

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="worker,planner,critic,full")
ap.add_argument("--limit", type=int, default=20)
ap.add_argument("--stub", action="store_true", help="use a fake model (no key, no cost)")
ap.add_argument("--verbose", action="store_true", help="print every event")
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
data_dir, results_dir = os.path.join(root, "data"), os.path.join(root, "results")
os.makedirs(results_dir, exist_ok=True)
tasks = json.load(open(os.path.join(data_dir, "tasks.json")))[:a.limit]
settings = Settings.from_env()

if a.stub:
    from agent_lab.stub import StubClient
    client = StubClient()
else:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("BASE_URL") or None)
agent = Agent(client, settings, base_dir=data_dir)


def show(e):
    d = e.to_dict(); d.pop("t"); kind = d.pop("type")
    print(f"    [{kind}] " + ", ".join(f"{k}={str(v)[:80]!r}" for k, v in d.items()))


log_path = os.path.join(results_dir, "results_stub.jsonl" if a.stub else "results.jsonl")
done = {}
if os.path.exists(log_path):
    for line in open(log_path):
        e = json.loads(line)
        done[(e["config"], e["id"])] = e
log, stopped = open(log_path, "a"), False
for name in a.configs.split(","):
    use_planner, use_critic = CONFIGS[name]
    for t in tasks:
        if (name, t["id"]) in done:
            continue
        try:
            res = agent.run(t, use_planner, use_critic, on_event=show if a.verbose else None)
        except Exception as ex:
            print(f"\nStopped at {name}/{t['id']}: {str(ex)[:200]}\nProgress is saved. Re-run the same command later to continue.")
            stopped = True
            break
        m = res.meter
        e = dict(config=name, id=t["id"], answer=res.answer, expected=t["expected"], ok=bool(check(t, res.answer)),
                 steps=m.steps, critic_calls=m.critic_calls, blocks=m.blocks, tokens=m.tokens, cost=m.cost(settings))
        done[(name, t["id"])] = e
        log.write(json.dumps(e) + "\n"); log.flush()
        print(f"{name:8}{t['id']:4}{'pass' if e['ok'] else 'FAIL'}")
    if stopped:
        break

summary = {}
for name in a.configs.split(","):
    es = [done[(name, t["id"])] for t in tasks if (name, t["id"]) in done]
    if not es:
        continue
    n = len(es)
    tier = collections.Counter(e["id"][0] for e in es if e["ok"])
    summary[name] = dict(n=n, passes=sum(e["ok"] for e in es), A=tier["A"], B=tier["B"], C=tier["C"],
                         steps=sum(e["steps"] for e in es) / n, tokens=sum(e["tokens"] for e in es) / n,
                         cost=sum(e["cost"] for e in es) / n)
print(f"\nModel: {settings.model}")
print(f"{'config':9}{'done':>6}{'passes':>8}{'A':>4}{'B':>4}{'C':>4}{'steps/task':>12}{'tokens/task':>13}{'$/task':>9}")
for k, s in summary.items():
    print(f"{k:9}{s['n']:>6}{s['passes']:>8}{s['A']:>4}{s['B']:>4}{s['C']:>4}{s['steps']:>12.1f}{s['tokens']:>13.0f}{s['cost']:>9.4f}")
if settings.price_in == 0:
    print("(PRICE_IN/PRICE_OUT not set: $/task is 0. On a free tier, report tokens/task as the cost measure.)")
if all(k in summary and summary[k]["n"] == len(tasks) for k in ("worker", "full")):
    w, f = summary["worker"], summary["full"]
    gain = f["passes"] - w["passes"]
    ratio = f["tokens"] / w["tokens"] if w["tokens"] else float("nan")
    print(f"\nGain (full - worker): {gain} tasks | full passes: {f['passes']} | tokens ratio: {ratio:.2f}x")
    print("SUCCESS (check cost bar separately if on a free tier)" if gain >= 4 and f["passes"] >= 14 else
          "WRONG (kill criterion)" if gain < 2 or ratio > 3 else "IN BETWEEN: tune the critic, re-run once")
else:
    print("\n(Verdict needs both 'worker' and 'full' finished on all tasks.)")
