from typing import Any, Optional

from pydantic import Field

from .base import (
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import (
    OpenAICompatibleModel,
    OpenAIRequestBody,
)


class MoonshotRequestBody(OpenAIRequestBody):
    # reference: https://platform.moonshot.cn/docs/api/chat
    ## excluded parameters
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)


@RemoteLanguageModel.register("moonshot")
class MoonshotModel(OpenAICompatibleModel):
    # reference: https://platform.moonshot.cn/docs/api-reference#list-models
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.moonshot.cn/v1/chat/completions",
        language_models=[
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
        ],
        visual_language_models=[],
        tool_models=[
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = MoonshotRequestBody
