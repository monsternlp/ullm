from typing import Any, Dict, Literal, Optional

from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    confloat,
)

from .base import GenerateConfig, JsonSchemaObject, RemoteLanguageModel, RemoteLanguageModelMetaInfo
from .openai import OpenAICompatibleModel, OpenAIRequestBody, OpenAIResponseBody


class BaiduStreamOptions(BaseModel):
    include_usage: Optional[bool] = None


class BaiduResponseFormat(BaseModel):
    type: Optional[Literal["text", "json_object", "json_schema"]] = "text"
    json_schema: Optional[JsonSchemaObject] = None


class BaiduWebSearchOptions(BaseModel):
    enable: Optional[bool] = None
    enable_citation: Optional[bool] = None
    enable_trace: Optional[bool] = None


class BaiduRequestBody(OpenAIRequestBody):
    # reference: https://platform.moonshot.cn/docs/api/chat
    ## excluded parameters
    logit_bias: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    max_tokens: Optional[Any] = Field(None, exclude=True)

    ## different parameters
    response_format: Optional[BaiduResponseFormat] = None

    ## baidu-specified parameters
    stream_options: Optional[BaiduStreamOptions] = None
    penalty_score: Optional[confloat(ge=1.0, le=2.0)] = None
    max_completion_tokens: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    web_search: Optional[BaiduWebSearchOptions] = None
    metadata: Optional[Dict[str, str]] = None


class BaiduSearchResult(BaseModel):
    index: int
    url: AnyUrl
    title: str


class BaiduPromptTokensDetails(BaseModel):
    search_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


class BaiduResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[BaiduPromptTokensDetails] = None


class BaiduResponseBody(OpenAIResponseBody):
    search_results: Optional[BaiduSearchResult] = None
    usage: BaiduResponseUsage


@RemoteLanguageModel.register("baidu")
class BaiduModel(OpenAICompatibleModel):
    # reference: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Fm2vrveyu
    META = RemoteLanguageModelMetaInfo(
        api_url="https://qianfan.baidubce.com/v2/chat/completions",
        language_models=[
            "ernie-4.0-8k-latest",
            "ernie-4.0-8k-preview",
            "ernie-4.0-8k",
            "ernie-4.0-turbo-8k-latest",
            "ernie-4.0-turbo-8k-preview",
            "ernie-4.0-turbo-8k",
            "ernie-4.0-turbo-128k",
            "ernie-3.5-8k-preview",
            "ernie-3.5-8k",
            "ernie-3.5-128k",
            "ernie-speed-8k",
            "ernie-speed-128k",
            "ernie-speed-pro-128k",
            "ernie-lite-8k",
            "ernie-lite-pro-128k",
            "ernie-tiny-8k",
            "ernie-char-8k",
            "ernie-char-fiction-8k",
            "ernie-novel-8k",
            "deepseek-v3",
            "deepseek-r1",
            "deepseek-r1-distill-qwen-32b",
            "deepseek-r1-distill-qwen-14b",
            "deepseek-r1-distill-qwen-7b",
            "deepseek-r1-distill-qwen-1.5b",
            "deepseek-r1-distill-llama-70b",
            "deepseek-r1-distill-llama-8b",
            "deepseek-r1-distill-qianfan-llama-70b",
            "deepseek-r1-distill-qianfan-llama-8b",
            "ernie-4.0-8k-latest-online",
            "ernie-4.0-8k-preview-online",
            "ernie-4.0-8k-online",
            "ernie-4.0-turbo-8k-latest-online",
            "ernie-4.0-turbo-8k-preview-online",
            "ernie-4.0-turbo-8k-online",
            "ernie-4.0-turbo-128k-online",
            "ernie-3.5-8k-preview-online",
            "ernie-3.5-8k-online",
            "ernie-3.5-128k-online",
        ],
        visual_language_models=[],
        tool_models=[
            "ernie-4.0-8k-latest",
            "ernie-4.0-8k-preview",
            "ernie-4.0-8k",
            "ernie-4.0-turbo-8k-latest",
            "ernie-4.0-turbo-8k-preview",
            "ernie-4.0-turbo-8k",
            "ernie-4.0-turbo-128k",
            "ernie-3.5-8k-preview",
            "ernie-3.5-8k",
            "ernie-3.5-128k",
            "ernie-speed-pro-128k",
            "ernie-lite-pro-128k",
            "ernie-4.0-8k-latest-online",
            "ernie-4.0-8k-preview-online",
            "ernie-4.0-8k-online",
            "ernie-4.0-turbo-8k-latest-online",
            "ernie-4.0-turbo-8k-preview-online",
            "ernie-4.0-turbo-8k-online",
            "ernie-4.0-turbo-128k-online",
            "ernie-3.5-8k-preview-online",
            "ernie-3.5-8k-online",
            "ernie-3.5-128k-online",
        ],
        online_models=[
            "ernie-4.0-8k-latest-online",
            "ernie-4.0-8k-preview-online",
            "ernie-4.0-8k-online",
            "ernie-4.0-turbo-8k-latest-online",
            "ernie-4.0-turbo-8k-preview-online",
            "ernie-4.0-turbo-8k-online",
            "ernie-4.0-turbo-128k-online",
            "ernie-3.5-8k-preview-online",
            "ernie-3.5-8k-online",
            "ernie-3.5-128k-online",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = BaiduRequestBody
    RESPONSE_BODY_CLS = BaiduResponseBody

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        generation_config = {
            "model": self.model.replace("-online", ""),
            "frequency_penalty": config.frequency_penalty,
            "max_completion_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "presence_penalty": config.presence_penalty,
            "response_format": {"type": config.response_format} if config.response_format else None,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
        }
        if self.is_online_model():
            generation_config["web_search"] = {
                "enable": True,
                "enable_citation": True,
                "enable_trace": True,
            }
        else:
            generation_config["web_search"] = {
                "enable": False,
            }

        return generation_config
