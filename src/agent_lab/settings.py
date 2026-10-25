"""All tunable numbers and names in ONE place. Nothing else in the code reads environment variables."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = "gemini-3.1-flash-lite"
    critic_model: str = ""        # empty means: use the same model as the worker
    price_in: float = 0.0         # $ per 1M input tokens (0 on a free tier)
    price_out: float = 0.0        # $ per 1M output tokens
    temperature: float | None = 0.0   # None means: do not send a temperature at all
    delay: float = 0.0            # seconds to wait after every model call (rate limits)
    max_tokens: int = 4000        # reply length limit per model call
    max_turns: int = 6            # tool calls the worker may make per step
    max_critic: int = 3           # critic calls allowed per task (budget)
    max_retries: int = 2          # fix/replan retries allowed per task (budget)
    stuck_after: int = 3          # failed code runs in a row before the worker is "stuck"
    api_attempts: int = 6         # tries per model call when the server errors

    @property
    def critic(self) -> str:
        return self.critic_model or self.model

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        e = os.environ if env is None else env
        temp = e.get("TEMPERATURE", "0")
        return cls(
            model=e.get("MODEL", cls.model),
            critic_model=e.get("CRITIC_MODEL", ""),
            price_in=float(e.get("PRICE_IN", 0) or 0),
            price_out=float(e.get("PRICE_OUT", 0) or 0),
            temperature=None if temp.strip().lower() == "none" else float(temp),
            delay=float(e.get("DELAY", 0) or 0),
        )
