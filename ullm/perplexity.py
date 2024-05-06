from typing import Any, Optional

from pydantic import Field, conint

from .base import (
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import OpenAICompatibleModel, OpenAIRequestBody


class PerplexityRequestBody(OpenAIRequestBody):
    # reference: https://docs.perplexity.ai/reference/post_chat_completions
    ## excluded parameters
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    tools: Optional[Any] = Field(None, exclude=True)
    tool_choice: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(None, exclude=True)
    ## Perplexity-specific parameters
    top_k: Optional[conint(ge=0, lt=2048)] = None


@RemoteLanguageModel.register("perplexity")
class PerplexityModel(OpenAICompatibleModel):
    # reference: https://docs.perplexity.ai/docs/model-cards
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.perplexity.ai/v1/chat/completions",
        language_models=[
            "llama-3-sonar-small-32k-chat",
            "llama-3-sonar-small-32k-online",
            "llama-3-sonar-large-32k-chat",
            "llama-3-sonar-large-32k-online",
            "llama-3-8b-instruct",
            "llama-3-70b-instruct",
            "mixtral-8x7b-instruct",
        ],
        visual_language_models=[],
        tool_models=[],
        online_models=[
            "llama-3-sonar-small-32k-online",
            "llama-3-sonar-large-32k-online",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = PerplexityRequestBody
