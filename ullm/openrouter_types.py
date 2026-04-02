from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .types import Thinking


class OpenRouterReasoning(BaseModel):
    """
    https://openrouter.ai/docs/use-cases/reasoning-tokens#reasoning-effort-level
    OpenRouter-only reasoning configuration.
    """

    effort: Annotated[
        Literal["xhigh", "high", "medium", "low", "minimal", "none"] | None,
        Field(description="OpenAI-style reasoning effort setting"),
    ] = None
    max_tokens: Annotated[
        int | None,
        Field(
            description="Non-OpenAI-style reasoning setting. Cannot be used together with effort."
        ),
    ] = None
    exclude: Annotated[
        bool | None, Field(description="Whether to exclude reasoning from the response")
    ] = False
    enabled: Annotated[bool | None, Field(description="Enable reasoning or not")] = None

    @classmethod
    def from_standard(cls, thinking: Thinking):
        if thinking.type == "enabled" and (thinking.effort or thinking.max_tokens > 0):
            return cls(
                effort=thinking.effort,
                max_tokens=thinking.max_tokens,
                exclude=thinking.exclude,
            )

        return cls(effort="none", exclude=True, enabled=False)
