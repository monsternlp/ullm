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


class ZeroOneAIRequestBody(OpenAIRequestBody):
    # https://platform.lingyiwanwu.com/docs#-%E5%85%A5%E5%8F%82%E6%8F%8F%E8%BF%B0
    ## excluded parameters
    frequency_penalty: Optional[Any] = Field(None, exclude=True)
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(None, exclude=True)
    presence_penalty: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    stop: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)


@RemoteLanguageModel.register("01ai")
class ZeroOneAIModel(OpenAICompatibleModel):
    # https://platform.lingyiwanwu.com/docs#模型与计费
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.lingyiwanwu.com/v1/chat/completions",
        language_models=[
            "yi-lightning",
            "yi-large",
            "yi-medium",
            "yi-medium-200k",
            "yi-spark",
            "yi-large-preview",
            "yi-large-fc",
        ],
        visual_language_models=[
            "yi-vision",
            "yi-vision-solution",
            "yi-vision-v2",
        ],
        tool_models=["yi-large-fc"],
        online_models=[],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = ZeroOneAIRequestBody
