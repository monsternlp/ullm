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


class DeepSeekRequestBody(OpenAIRequestBody):
    # reference: https://platform.moonshot.cn/docs/api/chat
    ## excluded parameters
    logit_bias: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    tools: Optional[Any] = Field(None, exclude=True)
    tool_choice: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)


@RemoteLanguageModel.register("deepseek")
class DeepSeekModel(OpenAICompatibleModel):
    # reference: https://platform.deepseek.com/api-docs/api/create-chat-completion/index.html
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.deepseek.com/v1/chat/completions",
        language_models=[
            "deepseek-chat",
            "deepseek-coder",
        ],
        visual_language_models=[],
        tool_models=[],
        online_models=[],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = DeepSeekRequestBody
