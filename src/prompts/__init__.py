"""Registry of input-prompt versions.

The evaluators build their VLM prompts through whichever version is selected, so a
prompt experiment is a new file here rather than an edit to the evaluation code:

    python run_refonbench_evaluation.py -cf cfg/eval_refonbench.yaml \
        --prompt-version history_included

Selection order: ``--prompt-version`` on the command line, else ``prompt_version`` in the
config, else ``default``.

Adding a version:
  1. create src/prompts/<name>.py with a PromptVersion subclass decorated
     @register("<name>")  (subclass DefaultPrompt to change only one of the prompts)
  2. import it at the bottom of this file so the decorator runs
"""

from typing import Dict, List, Type

from src.prompts.base import PromptVersion

DEFAULT_VERSION = "default"

_REGISTRY: Dict[str, Type[PromptVersion]] = {}


def register(name: str):
    """Class decorator that adds a PromptVersion to the registry."""

    def _decorator(cls: Type[PromptVersion]) -> Type[PromptVersion]:
        if name in _REGISTRY:
            raise ValueError(
                f"prompt version {name!r} is already registered by "
                f"{_REGISTRY[name].__module__}"
            )
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get(name: str = None) -> PromptVersion:
    """Instantiate a registered version. ``None``/empty selects the default."""
    name = name or DEFAULT_VERSION
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown prompt version {name!r}; available: {', '.join(available())}"
        )
    return _REGISTRY[name]()


def available() -> List[str]:
    """Registered version names, default first."""
    names = sorted(_REGISTRY)
    if DEFAULT_VERSION in names:
        names.remove(DEFAULT_VERSION)
        names.insert(0, DEFAULT_VERSION)
    return names


def describe() -> str:
    """Human-readable listing for --list-prompt-versions."""
    lines = []
    for name in available():
        mark = " (default)" if name == DEFAULT_VERSION else ""
        lines.append(f"  {name}{mark}: {_REGISTRY[name].description}")
    return "\n".join(lines)


# Importing the modules is what runs @register. Keep these at the bottom: the modules
# import `register` from this package, so the name has to exist first.
from src.prompts import default  # noqa: E402,F401
from src.prompts import history_included  # noqa: E402,F401
from src.prompts import action_rules  # noqa: E402,F401
