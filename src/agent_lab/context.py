"""Everything one run needs, bundled so the role functions take one argument."""
from dataclasses import dataclass, field


@dataclass
class RunContext:
    llm: object
    settings: object
    meter: object
    log: object
    base_dir: str
    history: list = field(default_factory=list)   # earlier successful code (see tools.run_code)
