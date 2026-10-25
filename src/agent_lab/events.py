"""Events: small records of what the agent did ("plan_created", "tool_call", ...).
A UI will later show these live; for now they are collected in a list and can be printed."""
import time
from dataclasses import dataclass, field


@dataclass
class Event:
    type: str
    data: dict
    t: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"type": self.type, "t": round(self.t, 3), **self.data}


class EventLog:
    def __init__(self, on_event=None):
        self.events: list[Event] = []
        self.on_event = on_event          # optional function called for every new event

    def emit(self, type: str, **data) -> Event:
        e = Event(type, data)
        self.events.append(e)
        if self.on_event:
            self.on_event(e)
        return e
