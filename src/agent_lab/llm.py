"""The only place that talks to the model API. Handles retries and counts tokens."""
import time


class LLM:
    def __init__(self, client, settings, meter, log=None, sleep=time.sleep):
        self.client, self.settings, self.meter, self.log, self.sleep = client, settings, meter, log, sleep

    def chat(self, system, messages, model=None, tools=None):
        """Send one request, return the reply message (with .content and maybe .tool_calls)."""
        s = self.settings
        kw = dict(model=model or s.model, max_tokens=s.max_tokens,
                  messages=[{"role": "system", "content": system}] + messages)
        if tools:
            kw["tools"] = tools
        if s.temperature is not None:
            kw["temperature"] = s.temperature
        for attempt in range(s.api_attempts):          # retry on rate limits / busy servers
            try:
                r = self.client.chat.completions.create(**kw)
                break
            except Exception as e:
                if attempt == s.api_attempts - 1:
                    raise
                if self.log:
                    self.log.emit("api_retry", attempt=attempt + 1, error=str(e)[:200])
                self.sleep(min(60, 5 * 2 ** attempt))
        u = r.usage
        self.meter.tin += getattr(u, "prompt_tokens", 0) or 0
        self.meter.tout += getattr(u, "completion_tokens", 0) or 0
        if s.delay:
            self.sleep(s.delay)
        return r.choices[0].message
