from typing import Annotated, Any, List, Optional

from pydantic import Field

from .base import (
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import OpenAICompatibleModel
from .openai_types import (
    OpenAIAssistantMessage,
    OpenAIRequestBody,
    OpenAIResponseBody,
    OpenAIResponseChoice,
)


class DeepSeekRequestBody(OpenAIRequestBody):
    # reference: https://platform.moonshot.cn/docs/api/chat
    ## excluded parameters
    logit_bias: Optional[Any] = Field(default=None, exclude=True)
    n: Optional[Any] = Field(default=None, exclude=True)
    seed: Optional[Any] = Field(default=None, exclude=True)
    user: Optional[Any] = Field(default=None, exclude=True)


class DeepSeekAssistantMessage(OpenAIAssistantMessage):
    # FIXME: prefix/reasoning_content 的使用需要 base_url="https://api.deepseek.com/beta"
    prefix: Optional[bool] = None


class DeepSeekResponseChoice(OpenAIResponseChoice):
    message: DeepSeekAssistantMessage


class DeepSeekResponseBody(OpenAIResponseBody):
    choices: Annotated[List[DeepSeekResponseChoice], Field(min_length=1)]

    def to_standard(self, model: str = None):
        result = super().to_standard(model=model)
        result.reasoning_content = self.choices[0].message.reasoning_content
        return result


@RemoteLanguageModel.register("deepseek")
class DeepSeekModel(OpenAICompatibleModel):
    # reference: https://platform.deepseek.com/api-docs/api/create-chat-completion/index.html
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.deepseek.com/chat/completions",
        language_models=[
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        visual_language_models=[],
        tool_models=["deepseek-chat"],
        online_models=[],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = DeepSeekRequestBody
    RESPONSE_BODY_CLS = DeepSeekResponseBody
