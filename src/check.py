"""check(task, answer) -> True/False. Use in your harness: passes = sum(check(t, agent(t)) for t in tasks)"""
import json, re
def check(task, answer):
    a, e = str(answer).strip(), task["expected"]
    if task["kind"] == "number":
        m = re.findall(r"-?\d[\d,]*\.?\d*", a)
        return bool(m) and abs(float(m[-1].replace(",", "")) - float(e)) <= task["tolerance"]
    norm = lambda s: re.sub(r"[^a-z0-9. ]", "", re.sub(r"\s+", " ", str(s).lower())).strip()
    return norm(e) in norm(a) if len(norm(e).split()) > 1 or norm(e).isalpha() else norm(e) == norm(a)
if __name__ == "__main__":
    tasks = json.load(open("tasks.json"))
    print("self-test (expected answers pass):", all(check(t, t["expected"]) for t in tasks))
