from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from .base import (
    GenerateConfig,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import (
    OpenAICompatibleModel,
    OpenAIRequestBody,
)


class ByteDanceThinkingOption(BaseModel):
    type: Optional[Literal["disabled", "enabled", "auto"]] = "auto"


class ByteDanceRequestBody(OpenAIRequestBody):
    # reference: https://www.volcengine.com/docs/82379/1298454
    ## ByteDance-specific parameters
    thinking: Optional[ByteDanceThinkingOption] = Field(None)

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
            "deepseek-r1-250120",
            "deepseek-r1-250528",
            "deepseek-r1-distill-qwen-32b-250120",
            "deepseek-r1-distill-qwen-7b-250120",
            "deepseek-v3-250324",
            "doubao-1-5-lite-32k-250115",
            "doubao-1-5-pro-256k-250115",
            "doubao-1-5-pro-32k-250115",
            "doubao-1-5-pro-32k-character-250228",
            "doubao-1-5-thinking-pro-250415",
            "doubao-1-5-thinking-pro-m-250415",
            "doubao-1-5-thinking-pro-m-250428",
            "doubao-lite-128k-240828",
            "doubao-lite-32k-240828",
            "doubao-lite-32k-character-241015",
            "doubao-lite-32k-character-250228",
            "doubao-lite-4k-character-240828",
            "doubao-pro-256k-241115",
            "doubao-pro-32k-240828",
            "doubao-pro-32k-241215",
            "doubao-pro-32k-character-241215",
            "doubao-pro-32k-character-240828",
            "doubao-pro-32k-browsing-240828",
            "doubao-pro-32k-browsing-241115",
            "doubao-pro-32k-functioncall-241028",
            "doubao-pro-32k-functioncall-preview",
        ],
        visual_language_models=[
            "doubao-1-5-vision-lite-250315",
            "doubao-1-5-vision-pro-250328",
            "doubao-1-5-vision-pro-32k-250115",
            "doubao-1-5-thinking-vision-pro-250428",
            "doubao-seed-1-6-250615",
            "doubao-seed-1-6-flash-250615",
            "doubao-seed-1-6-thinking-250615",
        ],
        online_models=[],
        tool_models=[
            "deepseek-r1-250120",
            "deepseek-r1-250528",
            "deepseek-r1-distill-qwen-32b-250120",
            "deepseek-r1-distill-qwen-7b-250120",
            "deepseek-v3-241226",
            "deepseek-v3-250324",
            "doubao-1-5-lite-32k-250115",
            "doubao-1-5-pro-256k-250115",
            "doubao-1-5-pro-32k-250115",
            "doubao-1-5-pro-32k-character-250228",
            "doubao-1-5-thinking-pro-250415",
            "doubao-1-5-thinking-pro-m-250415",
            "doubao-1-5-thinking-pro-m-250428",
            "doubao-1-5-vision-pro-250328",
            "doubao-1-5-vision-pro-32k-250115",
            "doubao-lite-128k-240828",
            "doubao-lite-32k-240828",
            "doubao-lite-32k-character-241015",
            "doubao-lite-32k-character-250228",
            "doubao-pro-256k-241115",
            "doubao-pro-32k-240828",
            "doubao-pro-32k-241215",
            "doubao-pro-32k-browsing-240828",
            "doubao-pro-32k-browsing-241115",
            "doubao-pro-32k-functioncall-241028",
            "doubao-pro-32k-functioncall-preview",
            "doubao-seed-1-6-250615",
            "doubao-seed-1-6-flash-250615",
            "doubao-seed-1-6-thinking-250615",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = ByteDanceRequestBody

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        generation_config = {
            "model": self.model,
            "frequency_penalty": config.frequency_penalty,
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "presence_penalty": config.presence_penalty,
            "response_format": {"type": config.response_format} if config.response_format else None,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
        }
        if config.thinking_type:
            if self.model in ("doubao-seed-1-6-250615", "doubao-1-5-thinking-vision-pro-250428"):
                generation_config["thinking"] = {"type": config.thinking_type}
            elif self.model in ("doubao-seed-1-6-flash-250615", "doubao-1-5-thinking-pro-m-250428"):
                generation_config["thinking"] = {
                    "type": config.thinking_type if config.thinking_type != "auto" else "disabled"
                }

        return generation_config
