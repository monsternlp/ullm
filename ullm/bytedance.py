from typing import Any, Dict, Optional

from pydantic import Field

from .base import (
    GenerateConfig,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import (
    OpenAICompatibleModel,
    OpenAIRequestBody,
)


class ByteDanceRequestBody(OpenAIRequestBody):
    # reference: https://www.volcengine.com/docs/82379/1298454
    ## excluded parameters
    n: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    tool_choice: Optional[Any] = Field(None, exclude=True)


@RemoteLanguageModel.register("bytedance")
class ByteDanceModel(OpenAICompatibleModel):
    # reference: https://www.volcengine.com/docs/82379/1298454
    META = RemoteLanguageModelMetaInfo(
        api_url="https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        language_models=[
            "Doubao-lite-4k",
            "Doubao-lite-32k",
            "Doubao-lite-128k",
            "Doubao-pro-4k",
            "Doubao-pro-32k",
            "Doubao-pro-128k",
            "Doubao-pro-256k",
            "Doubao-1.5-lite-32k",
            "Doubao-1.5-pro-32k",
            "Doubao-1.5-pro-256k",
        ],
        visual_language_models=[
            "Doubao-vision-lite-32k",
            "Doubao-vision-pro-32k",
            "Doubao-1.5-vision-pro-32k",
        ],
        online_models=[],
        tool_models=[
            "Doubao-lite-4k",
            "Doubao-lite-32k",
            "Doubao-lite-128k",
            "Doubao-pro-4k",
            "Doubao-pro-32k",
            "Doubao-pro-128k",
            "Doubao-pro-256k",
            "Doubao-1.5-lite-32k",
            "Doubao-1.5-pro-32k",
            "Doubao-1.5-pro-256k",
        ],
        required_config_fields=["api_key", "bytedance_endpoint"],
    )
    REQUEST_BODY_CLS = ByteDanceRequestBody

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "model": self.config.bytedance_endpoint,
            "frequency_penalty": config.frequency_penalty,
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "presence_penalty": config.presence_penalty,
            "response_format": {"type": config.response_format} if config.response_format else None,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
        }
