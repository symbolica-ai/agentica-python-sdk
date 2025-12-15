from textwrap import dedent
from typing import Any, Literal, overload

from agentica_internal.session_manager_messages import PromptTemplate


class template(str):
    _prompt_template: Literal[True] = True

    @staticmethod
    def is_template(value: Any) -> bool:
        return isinstance(value, template) or hasattr(value, '_prompt_template')

    def __new__(cls, value: str) -> "template":
        if not isinstance(value, str):
            raise TypeError("template must be a string")
        return super().__new__(cls, dedent(value).strip())

    def __str__(self) -> str:
        return '' + self

    def __repr__(self) -> str:
        return f'template({str(self)!r})'

    def format(self, *args: dict, **kwargs) -> str:
        result = str(self)
        for arg in args:
            if isinstance(arg, dict):
                kwargs = kwargs | arg
        for key, value in kwargs.items():
            if isinstance(key, str):
                result = result.replace('{{' + key + '}}', str(value))
        return result

    def to_prompt_template(self: str) -> PromptTemplate:
        return PromptTemplate(template=str(self))


@overload
def maybe_prompt_template(value: template) -> PromptTemplate: ...
@overload
def maybe_prompt_template(value: str) -> str | PromptTemplate: ...
def maybe_prompt_template(value: str) -> PromptTemplate | str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if template.is_template(value):
        return template.to_prompt_template(value)
    return value
