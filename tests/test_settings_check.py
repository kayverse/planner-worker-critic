from agent_lab.check import check
from agent_lab.settings import Settings


def test_settings_from_env():
    s = Settings.from_env({"MODEL": "m", "TEMPERATURE": "none", "PRICE_IN": "1", "DELAY": "6"})
    assert s.model == "m" and s.temperature is None and s.price_in == 1.0 and s.delay == 6.0 and s.critic == "m"


def test_check_numbers_and_text():
    num = {"kind": "number", "expected": 399.25, "tolerance": 0.01}
    assert check(num, "399.25") and not check(num, "399.5")
    word = {"kind": "text", "expected": "Mug", "tolerance": 0.01}
    assert check(word, "FINAL: mug") and not check(word, "Pen")
