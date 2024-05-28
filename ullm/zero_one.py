from typing import Optional

from pydantic import BaseModel, confloat, conlist

from .base import (
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import (
    OpenAIChatMessage,
    OpenAICompatibleModel,
)


class ZeroOneAIRequestBody(BaseModel):
    # https://platform.lingyiwanwu.com/docs#-%E5%85%A5%E5%8F%82%E6%8F%8F%E8%BF%B0
    messages: conlist(OpenAIChatMessage, min_length=1)
    model: str
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    temperature: Optional[confloat(ge=0.0, le=2.0)] = None
    top_p: Optional[confloat(ge=0.0, le=1.0)] = None


@RemoteLanguageModel.register("01ai")
class ZeroOneAIModel(OpenAICompatibleModel):
    # https://platform.lingyiwanwu.com/docs
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.lingyiwanwu.com/v1/chat/completions",
        language_models=[
            "yi-large",
            "yi-medium",
            "yi-medium-200k",
            "yi-spark",
            "yi-large-rag",
            "yi-large-turbo",
            "yi-large-preview",
            "yi-large-rag-preview",
        ],
        visual_language_models=["yi-vision"],
        tool_models=[],
        online_models=["yi-large-rag", "yi-large-rag-preview"],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = ZeroOneAIRequestBody
