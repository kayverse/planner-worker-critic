import pytest
from agent_lab import Agent, Settings
from fakes import ScriptedClient, boom, text, tool

TASK = {"question": "q", "data_file": "x.csv"}
ACCEPT = '{"verdict": "accept", "note": ""}'
FIX = '{"verdict": "fix", "note": "redo"}'


def make(replies, tmp_path, **settings):
    client = ScriptedClient(replies)
    return Agent(client, Settings(**settings), str(tmp_path), sleep=lambda s: None), client


def types(result):
    return [e["type"] for e in result.events]


def test_worker_only_happy_path(tmp_path):
    agent, _ = make([tool("print(2)"), text("FINAL: 2")], tmp_path)
    r = agent.run(TASK, use_planner=False, use_critic=False)
    assert r.answer == "2" and r.meter.steps == 1 and r.meter.tokens == 30
    assert types(r)[0] == "run_started" and types(r)[-1] == "run_finished" and "tool_call" in types(r)


def test_silent_reply_gets_one_nudge(tmp_path):
    agent, _ = make([tool("print(2)"), text(""), text("FINAL: 2")], tmp_path)
    r = agent.run(TASK, False, False)
    assert r.answer == "2" and "nudge" in types(r)


def test_planner_steps_run_in_order(tmp_path):
    agent, _ = make([text('["a", "b"]'), text("RESULT: x"), text("FINAL: 7")], tmp_path)
    r = agent.run(TASK, True, False)
    plan = next(e for e in r.events if e["type"] == "plan_created")
    assert plan["steps"] == ["a", "b"] and r.answer == "7"


def test_bad_plan_falls_back_to_one_step(tmp_path):
    agent, _ = make([text("not json at all"), text("FINAL: 3")], tmp_path)
    r = agent.run(TASK, True, False)
    assert next(e for e in r.events if e["type"] == "plan_created")["steps"] == ["q"] and r.answer == "3"


def test_critic_fix_then_accept(tmp_path):
    agent, _ = make([text("FINAL: 1"), text(FIX), text("FINAL: 2"), text(ACCEPT)], tmp_path)
    r = agent.run(TASK, False, True)
    assert r.answer == "2" and r.meter.critic_calls == 2
    assert [e["verdict"] for e in r.events if e["type"] == "critic_verdict"] == ["fix", "accept"]


def test_critic_retry_budget_is_respected(tmp_path):
    agent, _ = make([text("FINAL: 1"), text(FIX), text("FINAL: 2"), text(FIX), text("FINAL: 3")], tmp_path)
    r = agent.run(TASK, False, True)         # max_retries=2, so the third answer is not reviewed
    assert r.answer == "3" and r.meter.critic_calls == 2


def test_stuck_worker_calls_critic(tmp_path):
    bad = tool("raise ValueError('x')")
    agent, _ = make([bad, bad, bad, text(ACCEPT)], tmp_path)
    r = agent.run(TASK, False, True)
    assert "worker_stuck" in types(r) and r.meter.critic_calls == 1 and r.answer == "(step failed)"


def test_blocked_code_is_counted(tmp_path):
    agent, _ = make([tool("import subprocess"), text("FINAL: 1")], tmp_path)
    r = agent.run(TASK, False, False)
    assert r.meter.blocks == 1 and r.answer == "1"


def test_api_error_is_retried(tmp_path):
    agent, _ = make([boom("busy"), text("FINAL: 5")], tmp_path)
    r = agent.run(TASK, False, False)
    assert r.answer == "5" and "api_retry" in types(r)


def test_api_error_eventually_propagates(tmp_path):
    agent, _ = make([boom()] * 6, tmp_path)
    with pytest.raises(RuntimeError):
        agent.run(TASK, False, False)


def test_out_of_turns_forces_an_answer(tmp_path):
    agent, _ = make([tool("print(1)"), tool("print(2)"), text("FINAL: 9")], tmp_path, max_turns=2)
    r = agent.run(TASK, False, False)
    assert r.answer == "9" and "out_of_turns" in types(r)


def test_events_can_be_streamed(tmp_path):
    seen = []
    agent, _ = make([text("FINAL: 1")], tmp_path)
    agent.run(TASK, False, False, on_event=seen.append)
    assert [e.type for e in seen] == ["run_started", "step_started", "worker_reply", "run_finished"]


def test_answer_is_the_last_final_line(tmp_path):
    agent, _ = make([text('["a", "b"]'), text("FINAL: 1"), text("FINAL: 2")], tmp_path)
    assert agent.run(TASK, True, False).answer == "2"
