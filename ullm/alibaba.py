from typing import Any, Dict, Optional

from pydantic import (
    Field,
    conlist,
)

from .base import (
    GenerateConfig,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import OpenAIChatMessage, OpenAICompatibleModel, OpenAIRequestBody


class AlibabaRequestBody(OpenAIRequestBody):
    # reference: https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api
    messages: conlist(OpenAIChatMessage, min_length=1)

    ## exclude fields
    frequency_penalty: Optional[Any] = Field(None, exclude=True)
    tool_choice: Optional[Any] = Field(None, exclude=True)
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)

    ## Alibaba-specific parameters
    stream_options: Optional[dict] = None
    enable_search: Optional[bool] = None


@RemoteLanguageModel.register("alibaba")
class AlibabaModel(OpenAICompatibleModel):
    # https://help.aliyun.com/zh/model-studio/getting-started/models
    META = RemoteLanguageModelMetaInfo(
        api_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        language_models=[
            # noqa: https://help.aliyun.com/zh/model-studio/user-guide/text-generation
            "qwen-turbo",
            "qwen-turbo-latest",
            "qwen-turbo-1101",
            "qwen-turbo-0919",
            "qwen-turbo-0624",
            "qwen-turbo-0206",
            "qwen-plus",
            "qwen-plus-latest",
            "qwen-plus-0125",
            "qwen-plus-0916",
            "qwen-plus-0806",
            "qwen-plus-0723",
            "qwen-plus-0624",
            "qwen-plus-0206",
            "qwen-max",
            "qwen-max-latest",
            "qwen-max-0125",
            "qwen-max-0919",
            "qwen-max-0428",
            "qwen-max-0403",
            "qwen-max-0107",
            "qwen-turbo-online",
            "qwen-turbo-latest-online",
            "qwen-turbo-1101-online",
            "qwen-turbo-0919-online",
            "qwen-turbo-0624-online",
            "qwen-turbo-0206-online",
            "qwen-plus-online",
            "qwen-plus-latest-online",
            "qwen-plus-0125-online",
            "qwen-plus-0916-online",
            "qwen-plus-0806-online",
            "qwen-plus-0723-online",
            "qwen-plus-0624-online",
            "qwen-plus-0206-online",
            "qwen-max-online",
            "qwen-max-latest-online",
            "qwen-max-0125-online",
            "qwen-max-0919-online",
            "qwen-max-0428-online",
            "qwen-max-0403-online",
            "qwen-max-0107-online",
            # https://help.aliyun.com/zh/dashscope/developer-reference/qwen-long-api
            "qwen-long",
            # deepseek
            "deepseek-r1",
            "deepseek-v3",
            "deepseek-r1-distill-qwen-1.5b",
            "deepseek-r1-distill-qwen-7b",
            "deepseek-r1-distill-qwen-14b",
            "deepseek-r1-distill-qwen-32b",
            "deepseek-r1-distill-llama-8b",
            "deepseek-r1-distill-llama-70b",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/api-details-11
            "llama2-7b-chat-v2",
            "llama2-13b-chat-v2",
            "llama3-8b-instruct",
            "llama3-70b-instruct",
            "llama3.1-8b-instruct",
            "llama3.1-70b-instruct",
            "llama3.1-405b-instruct",
            "llama3.2-3b-instruct",
            "llama3.2-1b-instruct",
            "llama3.3-70b-instruct",
            # qwq
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-7b-14b-72b-api-detailes
            "qwq-32b-preview",
            "qwen2.5-14b-instruct-1m",
            "qwen2.5-7b-instruct-1m",
            "qwen2.5-72b-instruct",
            "qwen2.5-32b-instruct",
            "qwen2.5-14b-instruct",
            "qwen2.5-7b-instruct",
            "qwen2.5-3b-instruct",
            "qwen2.5-1.5b-instruct",
            "qwen2.5-0.5b-instruct",
            "qwen2-57b-a14b-instruct",
            "qwen2-72b-instruct",
            "qwen2-7b-instruct",
            "qwen2-1.5b-instruct",
            "qwen2-0.5b-instruct",
            "qwen1.5-110b-chat",
            "qwen1.5-72b-chat",
            "qwen1.5-32b-chat",
            "qwen1.5-14b-chat",
            "qwen1.5-7b-chat",
            "qwen1.5-1.8b-chat",
            "qwen1.5-0.5b-chat",
            "qwen-72b-chat",
            "qwen-14b-chat",
            "qwen-7b-chat",
            "qwen-1.8b-longcontext-chat",
            "qwen-1.8b-chat",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/api-details-2
            "baichuan-7b-v1",
            "baichuan2-7b-chat-v1",
            "baichuan2-13b-chat-v1",
            "baichuan2-turbo",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/api-details-8
            "chatglm-6b-v2",
            "chatglm3-6b",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/api-details-3
            "chatyuan-large-v2",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/yi-series-models-api-details
            "yi-large",
            "yi-medium",
            "yi-large-rag",
            "yi-large-turbo",
            # minimax
            "abab6.5g-chat",
            "abab6.5t-chat",
            "abab6.5s-chat",
        ],
        visual_language_models=[
            "qwen-omni-turbo",
            "qwen-omni-turbo-latest",
            "qwen-omni-turbo-0119",
            # noqa: https://help.aliyun.com/zh/model-studio/user-guide/vision#f1cbd5b8a8k5w
            "qwen-vl-plus",
            "qwen-vl-plus-0125",
            "qwen-vl-plus-0102",
            "qwen-vl-plus-0809",
            "qwen-vl-plus-1201",
            "qwen-vl-plus-latest",
            "qwen-vl-ocr",
            "qwen-vl-ocr-1028",
            "qwen-vl-ocr-latest",
            "qwen-vl-max",
            "qwen-vl-max-0125",
            "qwen-vl-max-1230",
            "qwen-vl-max-1119",
            "qwen-vl-max-1030",
            "qwen-vl-max-0809",
            "qwen-vl-max-0201",
            "qwen-vl-max-latest",
            # 2024-10-07 目前仅供免费体验，免费额度用完后不可调用
            "qwen-vl-v1",
            "qwen-vl-chat-v1",
            "qwen2-vl-72b-instruct",
            "qwen2-vl-7b-instruct",
            "qwen2-vl-2b-instruct",
            "qvq-72b-preview",
            "qwen2.5-vl-3b-instruct",
            "llama3.2-90b-vision-instruct",
            "llama3.2-11b-vision",
        ],
        tool_models=[
            "qwen-turbo",
            "qwen-turbo-latest",
            "qwen-turbo-1101",
            "qwen-turbo-0919",
            "qwen-turbo-0624",
            "qwen-turbo-0206",
            "qwen-plus",
            "qwen-plus-latest",
            "qwen-plus-0125",
            "qwen-plus-0916",
            "qwen-plus-0806",
            "qwen-plus-0723",
            "qwen-plus-0624",
            "qwen-plus-0206",
            "qwen-max",
            "qwen-max-latest",
            "qwen-max-0125",
            "qwen-max-0919",
            "qwen-max-0428",
            "qwen-max-0403",
            "qwen-max-0107",
            "qwen-max-longcontext",
        ],
        online_models=[
            "qwen-turbo-online",
            "qwen-turbo-latest-online",
            "qwen-turbo-1101-online",
            "qwen-turbo-0919-online",
            "qwen-turbo-0624-online",
            "qwen-turbo-0206-online",
            "qwen-plus-online",
            "qwen-plus-latest-online",
            "qwen-plus-0125-online",
            "qwen-plus-0916-online",
            "qwen-plus-0806-online",
            "qwen-plus-0723-online",
            "qwen-plus-0624-online",
            "qwen-plus-0206-online",
            "qwen-max-online",
            "qwen-max-latest-online",
            "qwen-max-0125-online",
            "qwen-max-0919-online",
            "qwen-max-0428-online",
            "qwen-max-0403-online",
            "qwen-max-0107-online",
            "yi-large-rag",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = AlibabaRequestBody

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        json_models = [
            "qwen-max-0919",
            "qwen-max-latest",
            "qwen-plus",
            "qwen-plus-0919",
            "qwen-plus-latest",
            "qwen-turbo",
            "qwen-turbo-1101",
            "qwen-turbo-0919",
            "qwen-turbo-latest",
            "qwen2.5-72b-instruct",
            "qwen2.5-32b-instruct",
            "qwen2.5-14b-instruct",
            "qwen2.5-7b-instruct",
            "qwen2.5-3b-instruct",
            "qwen2.5-1.5b-instruct",
            "qwen2.5-0.5b-instruct",
        ]
        generation_config = {
            "model": self.model.replace("-online", ""),
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "presence_penalty": config.presence_penalty,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
            "enable_search": self.is_online_model() or None,
        }
        if generation_config["model"] in json_models and config.response_format == "json_object":
            generation_config["response_format"] = {"type": "json_object"}

        return generation_config
