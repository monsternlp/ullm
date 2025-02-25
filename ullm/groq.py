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
            "gemma2-9b-it",
            "llama-3.3-70b-versatile",
            "llama-3.3-70b-specdec",
            "llama-3.1-8b-instant",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "llama-guard-3-8b",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "qwen-2.5-coder-32b",
            "qwen-2.5-32b",
            "deepseek-r1-distill-qwen-32b",
            "deepseek-r1-distill-llama-70b-specdec",
            "deepseek-r1-distill-llama-70b",
        ],
        visual_language_models=[
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
        ],
        tool_models=[
            "gemma2-9b-it",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "qwen-2.5-coder-32b",
            "deepseek-r1-distill-qwen-32b",
            "deepseek-r1-distill-llama-70b",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = GroqRequestBody
