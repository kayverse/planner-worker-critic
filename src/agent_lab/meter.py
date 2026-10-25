"""Counts what a run used: tokens, tool calls, critic calls, blocked code."""
from dataclasses import dataclass


@dataclass
class Meter:
    tin: int = 0            # input tokens
    tout: int = 0           # output tokens
    steps: int = 0          # code runs by the worker
    critic_calls: int = 0
    blocks: int = 0         # code stopped by the validator

    @property
    def tokens(self) -> int:
        return self.tin + self.tout

    def cost(self, settings) -> float:
        return (self.tin * settings.price_in + self.tout * settings.price_out) / 1e6
