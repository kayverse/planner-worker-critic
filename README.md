# Planner + Worker + Critic: does each role earn its place?

A small agent that answers questions about CSV sales data (mock data only), built from three LLM roles
plus a rule-based validator, and a test harness that measures whether the planner and critic actually help.

## Result so far (1 run, 20 tasks, gemini-3.1-flash-lite free tier, temperature 0)
| Setup | Correct | Steps / task | Tokens / task |
|---|---|---|---|
| Worker only | 19 of 20 | 3.8 | 6,926 |
| Planner + Worker + Critic | 17 of 20 | 13.8 | 23,494 (3.39x) |

The hypothesis (full system beats worker by at least 4 tasks) was rejected: the gain was -2 tasks.
Planner-only and critic-only runs are still to do. Limits: one run, small sample, one small model.

## How it works
Question -> Planner (subtasks) -> Validator (blocks unsafe code) -> Worker (runs pandas code) -> Critic
(only when stuck or on the final answer; max 3 calls, 2 retries) -> answer.
Every step is reported as an event (`plan_created`, `tool_call`, `critic_verdict`, ...), which a UI can show live.

## Setup (Windows, Command Prompt)
1. `python -m venv .venv`
2. `.venv\Scripts\activate`
3. `pip install -r requirements-dev.txt`
4. Copy `.env.example` to `.env` and paste your own API key into it.
5. Run the tests (no key needed, no cost): `python -m pytest`
6. Offline check with a fake model: `python src\run.py --stub`
7. Real run: `python src\run.py --configs worker --limit 3`, then `python src\run.py`
   (add `--verbose` to watch every event)

## Folders
- `src/agent_lab/` the agent: `settings` (all numbers), `llm` (API calls + retries), `tools` (validator + code runner),
  `roles` (planner, worker, critic), `runner` (the control loop), `events`, `meter`, `check` (answer checker), `stub`
- `src/run.py` command-line test runner
- `tests/` automatic tests using a scripted fake model
- `data/` mock CSVs, the 20 tasks (`tasks.json`) and the script that makes them (`make_tasks.py`)
- `results/` saved results (`results.jsonl`), one line per task

## Safety note
The model's Python currently runs on your own computer, guarded only by simple text rules. Use mock data only.
Sandboxing is on the to-do list.
