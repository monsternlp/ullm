from typing import Any, Optional

from pydantic import Field

from .base import (
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import OpenAICompatibleModel, OpenAIRequestBody


class GroqRequestBody(OpenAIRequestBody):
    # https://console.groq.com/docs/text-chat
    # https://console.groq.com/docs/openai
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(None, exclude=True)


@RemoteLanguageModel.register("groq")
class GroqModel(OpenAICompatibleModel):
    # reference: https://console.groq.com/docs/models
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.groq.com/openai/v1/chat/completions",
        language_models=[
            "lama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
        ],
        visual_language_models=[],
        tool_models=[
            "lama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = GroqRequestBody
