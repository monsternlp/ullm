from typing import Any, Literal, Optional

from pydantic import Field, conint

from .base import (
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import OpenAICompatibleModel, OpenAIRequestBody


class PerplexityRequestBody(OpenAIRequestBody):
    # reference: https://docs.perplexity.ai/api-reference/chat-completions
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
    search_domain_filter: Optional[list] = None
    return_images: Optional[bool] = None
    return_related_questions: Optional[bool] = None
    search_recency_filter: Optional[Literal["month", "week", "day", "hour"]] = None


@RemoteLanguageModel.register("perplexity")
class PerplexityModel(OpenAICompatibleModel):
    # reference: https://docs.perplexity.ai/docs/model-cards
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.perplexity.ai/chat/completions",
        language_models=[
            "llama-3.1-sonar-small-128k-chat",
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-chat",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-huge-128k-online",
            "llama-3.1-8b-instruct",
            "llama-3.1-70b-instruct",
        ],
        visual_language_models=[],
        tool_models=[],
        online_models=[
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-huge-128k-online",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = PerplexityRequestBody
