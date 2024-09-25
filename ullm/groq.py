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
            "gemma-7b-it",
            "gemma2-9b-it",
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-11b-text-preview",
            "llama-3.2-90b-text-preview",
            "llama-guard-3-8b",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "llama3-groq-8b-8192-tool-use-preview",
            "llama3-groq-70b-8192-tool-use-preview",
            "mixtral-8x7b-32768",
        ],
        visual_language_models=[
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
        ],
        tool_models=[
            "gemma-7b-it",
            "gemma2-9b-it",
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "llama3-groq-8b-8192-tool-use-preview",
            "llama3-groq-70b-8192-tool-use-preview",
            "mixtral-8x7b-32768",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = GroqRequestBody
