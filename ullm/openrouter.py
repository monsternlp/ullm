from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, confloat, conint, conlist

from .base import (
    GenerateConfig,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import (
    OpenAIAssistantMessage,
    OpenAICompatibleModel,
    OpenAIRequestBody,
    OpenAIResponseBody,
)


class OpenRouterProviderSetting(BaseModel):
    # https://openrouter.ai/docs#provider-routing
    order: Optional[
        List[
            Literal[
                "OpenAI",
                "Anthropic",
                "HuggingFace",
                "Google",
                "Mancer",
                "Mancer 2",
                "Together",
                "DeepInfra",
                "Azure",
                "Modal",
                "AnyScale",
                "Replicate",
                "Perplexity",
                "Recursal",
                "Fireworks",
                "Mistral",
                "Groq",
                "Cohere",
                "Lepton",
                "OctoAI",
                "Novita",
                "Lynn",
                "Lynn 2",
            ]
        ]
    ] = None
    require_parameters: Optional[bool] = None
    data_collection: Optional[Literal["deny", "allow"]] = None
    allow_fallbacks: Optional[bool] = None


class OpenRouterRequestBody(OpenAIRequestBody):
    # reference: https://openrouter.ai/docs#requests
    # OpenAI 没有的参数
    prompt: Optional[str] = None
    top_k: Optional[conint(ge=1)] = None
    repetition_penalty: Optional[confloat(gt=0.0, le=2.0)] = None
    transforms: Optional[List[Literal["middle-out"]]] = None
    ## models routing
    provider: Optional[OpenRouterProviderSetting] = None
    models: Optional[List[str]] = None
    route: Optional[Literal["fallback"]] = None


class OpenRouterResponseChoice(BaseModel):
    finish_reason: Optional[str] = None
    index: int
    message: OpenAIAssistantMessage


class OpenRouterResponseBody(OpenAIResponseBody):
    choices: conlist(OpenRouterResponseChoice, min_length=1)


@RemoteLanguageModel.register("openrouter")
class OpenRouterModel(OpenAICompatibleModel):
    # https://openrouter.ai/api/v1/models
    _FULL_SUPPORTED_MODELS = [
        "aetherwiing/mn-starcannon-12b",
        "ai21/jamba-1-5-large",
        "ai21/jamba-1-5-mini",
        "ai21/jamba-instruct",
        "alpindale/goliath-120b",
        "alpindale/magnum-72b",
        "anthracite-org/magnum-v2-72b",
        "anthracite-org/magnum-v4-72b",
        "anthropic/claude-2",
        "anthropic/claude-2.0",
        "anthropic/claude-2.0:beta",
        "anthropic/claude-2.1",
        "anthropic/claude-2.1:beta",
        "anthropic/claude-2:beta",
        "anthropic/claude-3-5-haiku",
        "anthropic/claude-3-5-haiku-20241022",
        "anthropic/claude-3-5-haiku-20241022:beta",
        "anthropic/claude-3-5-haiku:beta",
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-haiku:beta",
        "anthropic/claude-3-opus",
        "anthropic/claude-3-opus:beta",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-sonnet:beta",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.5-sonnet-20240620",
        "anthropic/claude-3.5-sonnet-20240620:beta",
        "anthropic/claude-3.5-sonnet:beta",
        "cognitivecomputations/dolphin-mixtral-8x22b",
        "cognitivecomputations/dolphin-mixtral-8x7b",
        "cohere/command",
        "cohere/command-r",
        "cohere/command-r-03-2024",
        "cohere/command-r-08-2024",
        "cohere/command-r-plus",
        "cohere/command-r-plus-04-2024",
        "cohere/command-r-plus-08-2024",
        "databricks/dbrx-instruct",
        "deepseek/deepseek-chat",
        "eva-unit-01/eva-qwen-2.5-14b",
        "eva-unit-01/eva-qwen-2.5-32b",
        "google/gemini-flash-1.5",
        "google/gemini-flash-1.5-8b",
        "google/gemini-flash-1.5-8b-exp",
        "google/gemini-flash-1.5-exp",
        "google/gemini-pro",
        "google/gemini-pro-1.5",
        "google/gemini-pro-1.5-exp",
        "google/gemini-pro-vision",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-9b-it:free",
        "google/palm-2-chat-bison",
        "google/palm-2-chat-bison-32k",
        "google/palm-2-codechat-bison",
        "google/palm-2-codechat-bison-32k",
        "gryphe/mythomax-l2-13b",
        "gryphe/mythomax-l2-13b:extended",
        "gryphe/mythomax-l2-13b:free",
        "gryphe/mythomax-l2-13b:nitro",
        "gryphe/mythomist-7b",
        "gryphe/mythomist-7b:free",
        "huggingfaceh4/zephyr-7b-beta:free",
        "inflection/inflection-3-pi",
        "inflection/inflection-3-productivity",
        "jondurbin/airoboros-l2-70b",
        "liquid/lfm-40b",
        "liquid/lfm-40b:free",
        "lizpreciatior/lzlv-70b-fp16-hf",
        "mancer/weaver",
        "meta-llama/llama-2-13b-chat",
        "meta-llama/llama-3-70b-instruct",
        "meta-llama/llama-3-70b-instruct:nitro",
        "meta-llama/llama-3-8b-instruct",
        "meta-llama/llama-3-8b-instruct:extended",
        "meta-llama/llama-3-8b-instruct:free",
        "meta-llama/llama-3-8b-instruct:nitro",
        "meta-llama/llama-3.1-405b",
        "meta-llama/llama-3.1-405b-instruct",
        "meta-llama/llama-3.1-405b-instruct:free",
        "meta-llama/llama-3.1-405b-instruct:nitro",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-70b-instruct:free",
        "meta-llama/llama-3.1-70b-instruct:nitro",
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-8b-instruct:free",
        "meta-llama/llama-3.2-11b-vision-instruct",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "meta-llama/llama-3.2-1b-instruct",
        "meta-llama/llama-3.2-1b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.2-3b-instruct:free",
        "meta-llama/llama-3.2-90b-vision-instruct",
        "meta-llama/llama-3.2-90b-vision-instruct:free",
        "meta-llama/llama-guard-2-8b",
        "microsoft/phi-3-medium-128k-instruct",
        "microsoft/phi-3-medium-128k-instruct:free",
        "microsoft/phi-3-mini-128k-instruct",
        "microsoft/phi-3-mini-128k-instruct:free",
        "microsoft/phi-3.5-mini-128k-instruct",
        "microsoft/wizardlm-2-7b",
        "microsoft/wizardlm-2-8x22b",
        "mistralai/codestral-mamba",
        "mistralai/ministral-3b",
        "mistralai/ministral-8b",
        "mistralai/mistral-7b-instruct",
        "mistralai/mistral-7b-instruct-v0.1",
        "mistralai/mistral-7b-instruct-v0.2",
        "mistralai/mistral-7b-instruct-v0.3",
        "mistralai/mistral-7b-instruct:free",
        "mistralai/mistral-7b-instruct:nitro",
        "mistralai/mistral-large",
        "mistralai/mistral-medium",
        "mistralai/mistral-nemo",
        "mistralai/mistral-small",
        "mistralai/mistral-tiny",
        "mistralai/mixtral-8x22b-instruct",
        "mistralai/mixtral-8x7b",
        "mistralai/mixtral-8x7b-instruct",
        "mistralai/mixtral-8x7b-instruct:nitro",
        "mistralai/pixtral-12b",
        "neversleep/llama-3-lumimaid-70b",
        "neversleep/llama-3-lumimaid-8b",
        "neversleep/llama-3-lumimaid-8b:extended",
        "neversleep/llama-3.1-lumimaid-70b",
        "neversleep/llama-3.1-lumimaid-8b",
        "neversleep/noromaid-20b",
        "nothingiisreal/mn-celeste-12b",
        "nousresearch/hermes-2-pro-llama-3-8b",
        "nousresearch/hermes-2-theta-llama-3-8b",
        "nousresearch/hermes-3-llama-3.1-405b",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "nousresearch/hermes-3-llama-3.1-70b",
        "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
        "nousresearch/nous-hermes-llama2-13b",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "openai/chatgpt-4o-latest",
        "openai/gpt-3.5-turbo",
        "openai/gpt-3.5-turbo-0125",
        "openai/gpt-3.5-turbo-0613",
        "openai/gpt-3.5-turbo-1106",
        "openai/gpt-3.5-turbo-16k",
        "openai/gpt-3.5-turbo-instruct",
        "openai/gpt-4",
        "openai/gpt-4-0314",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-32k",
        "openai/gpt-4-32k-0314",
        "openai/gpt-4-turbo",
        "openai/gpt-4-turbo-preview",
        "openai/gpt-4-vision-preview",
        "openai/gpt-4o",
        "openai/gpt-4o-2024-05-13",
        "openai/gpt-4o-2024-08-06",
        "openai/gpt-4o-mini",
        "openai/gpt-4o-mini-2024-07-18",
        "openai/gpt-4o:extended",
        "openai/o1-mini",
        "openai/o1-mini-2024-09-12",
        "openai/o1-preview",
        "openai/o1-preview-2024-09-12",
        "openchat/openchat-7b",
        "openchat/openchat-7b:free",
        "openrouter/auto",
        "perplexity/llama-3-sonar-large-32k-chat",
        "perplexity/llama-3-sonar-large-32k-online",
        "perplexity/llama-3-sonar-small-32k-chat",
        "perplexity/llama-3.1-sonar-huge-128k-online",
        "perplexity/llama-3.1-sonar-large-128k-chat",
        "perplexity/llama-3.1-sonar-large-128k-online",
        "perplexity/llama-3.1-sonar-small-128k-chat",
        "perplexity/llama-3.1-sonar-small-128k-online",
        "pygmalionai/mythalion-13b",
        "qwen/qwen-2-72b-instruct",
        "qwen/qwen-2-7b-instruct",
        "qwen/qwen-2-7b-instruct:free",
        "qwen/qwen-2-vl-72b-instruct",
        "qwen/qwen-2-vl-7b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwen-2.5-7b-instruct",
        "raifle/sorcererlm-8x22b",
        "sao10k/fimbulvetr-11b-v2",
        "sao10k/l3-euryale-70b",
        "sao10k/l3-lunaris-8b",
        "sao10k/l3.1-euryale-70b",
        "sophosympatheia/midnight-rose-70b",
        "teknium/openhermes-2.5-mistral-7b",
        "thedrummer/rocinante-12b",
        "undi95/remm-slerp-l2-13b",
        "undi95/remm-slerp-l2-13b:extended",
        "undi95/toppy-m-7b",
        "undi95/toppy-m-7b:free",
        "undi95/toppy-m-7b:nitro",
        "x-ai/grok-beta",
        "xwin-lm/xwin-lm-70b",
    ]
    _MODEL_PREFIX_MAP = dict(
        [full_model_name.split("/")[::-1] for full_model_name in _FULL_SUPPORTED_MODELS]
    )
    _SUPPORTED_MODELS = sorted(_MODEL_PREFIX_MAP.keys())
    _VISUAL_MODELS = [
        "claude-3-haiku",
        "claude-3-haiku:beta",
        "claude-3-opus",
        "claude-3-opus:beta",
        "claude-3-sonnet",
        "claude-3-sonnet:beta",
        "claude-3.5-sonnet",
        "claude-3.5-sonnet-20240620",
        "claude-3.5-sonnet-20240620:beta",
        "claude-3.5-sonnet:beta",
        "gemini-flash-1.5",
        "gemini-flash-1.5-8b",
        "gemini-flash-1.5-8b-exp",
        "gemini-flash-1.5-exp",
        "gemini-pro-1.5",
        "gemini-pro-1.5-exp",
        "gemini-pro-vision",
        "llama-3.2-11b-vision-instruct",
        "llama-3.2-11b-vision-instruct:free",
        "llama-3.2-90b-vision-instruct",
        "llama-3.2-90b-vision-instruct:free",
        "pixtral-12b",
        "chatgpt-4o-latest",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o:extended",
        "qwen-2-vl-72b-instruct",
        "qwen-2-vl-7b-instruct",
    ]
    META = RemoteLanguageModelMetaInfo(
        api_url="https://openrouter.ai/api/v1/chat/completions",
        required_config_fields=["api_key"],
        language_models=set(_SUPPORTED_MODELS) - set(_VISUAL_MODELS),
        visual_language_models=_VISUAL_MODELS,
        tool_models=_SUPPORTED_MODELS,
        online_models=[
            "llama-3-sonar-large-32k-online",
            "llama-3.1-sonar-huge-128k-online",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-small-128k-online",
        ],
    )
    REQUEST_BODY_CLS = OpenRouterRequestBody
    RESPONSE_BODY_CLS = OpenRouterResponseBody

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        model_prefix = self._MODEL_PREFIX_MAP[self.model]
        return {
            "model": f"{model_prefix}/{self.model}",
            "frequency_penalty": config.frequency_penalty,
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "presence_penalty": config.presence_penalty,
            "response_format": {"type": config.response_format} if config.response_format else None,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
            "top_k": config.top_k or self.config.top_k,
            "repetition_penalty": config.repetition_penalty,
        }
