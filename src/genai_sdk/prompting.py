"""Prompt template utilities."""

from dataclasses import dataclass
from string import Formatter

from .errors import ConfigurationError


@dataclass(slots=True)
class PromptTemplate:
    """String template with strict variable checking."""

    template: str

    def variables(self) -> set[str]:
        vars_: set[str] = set()
        for _, field_name, _, _ in Formatter().parse(self.template):
            if field_name:
                vars_.add(field_name)
        return vars_

    def render(self, **kwargs: str) -> str:
        required = self.variables()
        missing = sorted(required.difference(kwargs))
        if missing:
            raise ConfigurationError(f"Missing template variables: {missing}")
        return self.template.format(**kwargs)
