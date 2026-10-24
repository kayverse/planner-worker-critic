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
Question -> Planner (subtasks) -> Validator (blocks unsafe code) -> Worker (runs pandas code) -> Critic (only when stuck or on the final answer; max 3 calls, 2 retries) -> answer.

## Setup (Windows, Command Prompt)
1. `python -m venv .venv`
2. `.venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and paste your own API key into it.
5. Offline check (fake model, no key needed): `python src\run.py --stub`
6. Real run: `python src\run.py --configs worker --limit 3`, then `python src\run.py`

## Folders
- `src/` the agent (`agent.py`), the test runner (`run.py`), the answer checker (`check.py`), a fake model (`stub.py`)
- `data/` mock CSVs, the 20 tasks (`tasks.json`) and the script that makes them (`make_tasks.py`)
- `results/` saved results (`results.jsonl`), one line per task

## Safety note
The model's Python currently runs on your own computer, guarded only by simple text rules. Use mock data only. Sandboxing is on the to-do list.
