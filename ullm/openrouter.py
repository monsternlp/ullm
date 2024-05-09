from typing import List, Literal, Optional

from pydantic import BaseModel, confloat, conint, conlist

from .base import (
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
        "01-ai/yi-34b",
        "01-ai/yi-34b-chat",
        "01-ai/yi-6b",
        "alpindale/goliath-120b",
        "anthropic/claude-2",
        "anthropic/claude-2.0",
        "anthropic/claude-2.0:beta",
        "anthropic/claude-2.1",
        "anthropic/claude-2.1:beta",
        "anthropic/claude-2:beta",
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-haiku:beta",
        "anthropic/claude-3-opus",
        "anthropic/claude-3-opus:beta",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-sonnet:beta",
        "anthropic/claude-instant-1",
        "anthropic/claude-instant-1:beta",
        "austism/chronos-hermes-13b",
        "codellama/codellama-70b-instruct",
        "cognitivecomputations/dolphin-mixtral-8x7b",
        "cohere/command",
        "cohere/command-r",
        "cohere/command-r-plus",
        "databricks/dbrx-instruct",
        "fireworks/firellava-13b",
        "google/gemini-pro",
        "google/gemini-pro-1.5",
        "google/gemini-pro-vision",
        "google/gemma-7b-it",
        "google/gemma-7b-it:free",
        "google/gemma-7b-it:nitro",
        "google/palm-2-chat-bison",
        "google/palm-2-chat-bison-32k",
        "google/palm-2-codechat-bison",
        "google/palm-2-codechat-bison-32k",
        "gryphe/mythomax-l2-13b",
        "gryphe/mythomax-l2-13b:extended",
        "gryphe/mythomax-l2-13b:nitro",
        "gryphe/mythomist-7b",
        "gryphe/mythomist-7b:free",
        "haotian-liu/llava-13b",
        "huggingfaceh4/zephyr-7b-beta:free",
        "intel/neural-chat-7b",
        "jondurbin/airoboros-l2-70b",
        "koboldai/psyfighter-13b-2",
        "lizpreciatior/lzlv-70b-fp16-hf",
        "lynn/soliloquy-l3",
        "mancer/weaver",
        "meta-llama/codellama-34b-instruct",
        "meta-llama/llama-2-13b-chat",
        "meta-llama/llama-2-70b-chat",
        "meta-llama/llama-2-70b-chat:nitro",
        "meta-llama/llama-3-70b-instruct",
        "meta-llama/llama-3-70b-instruct:nitro",
        "meta-llama/llama-3-8b-instruct",
        "meta-llama/llama-3-8b-instruct:extended",
        "meta-llama/llama-3-8b-instruct:nitro",
        "microsoft/wizardlm-2-7b",
        "microsoft/wizardlm-2-8x22b",
        "microsoft/wizardlm-2-8x22b:nitro",
        "mistralai/mistral-7b-instruct",
        "mistralai/mistral-7b-instruct:free",
        "mistralai/mistral-7b-instruct:nitro",
        "mistralai/mistral-large",
        "mistralai/mistral-medium",
        "mistralai/mistral-small",
        "mistralai/mistral-tiny",
        "mistralai/mixtral-8x22b",
        "mistralai/mixtral-8x22b-instruct",
        "mistralai/mixtral-8x7b",
        "mistralai/mixtral-8x7b-instruct",
        "mistralai/mixtral-8x7b-instruct:nitro",
        "neversleep/llama-3-lumimaid-8b",
        "neversleep/llama-3-lumimaid-8b:extended",
        "neversleep/noromaid-20b",
        "neversleep/noromaid-mixtral-8x7b-instruct",
        "nousresearch/nous-capybara-34b",
        "nousresearch/nous-capybara-7b",
        "nousresearch/nous-capybara-7b:free",
        "nousresearch/nous-hermes-2-mistral-7b-dpo",
        "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
        "nousresearch/nous-hermes-2-mixtral-8x7b-sft",
        "nousresearch/nous-hermes-2-vision-7b",
        "nousresearch/nous-hermes-llama2-13b",
        "nousresearch/nous-hermes-yi-34b",
        "open-orca/mistral-7b-openorca",
        "openai/gpt-3.5-turbo",
        "openai/gpt-3.5-turbo-0125",
        "openai/gpt-3.5-turbo-16k",
        "openai/gpt-3.5-turbo-instruct",
        "openai/gpt-4",
        "openai/gpt-4-32k",
        "openai/gpt-4-turbo",
        "openai/gpt-4-turbo-preview",
        "openai/gpt-4-vision-preview",
        "openchat/openchat-7b",
        "openchat/openchat-7b:free",
        "openrouter/auto",
        "openrouter/cinematika-7b",
        "openrouter/cinematika-7b:free",
        "perplexity/pplx-70b-chat",
        "perplexity/pplx-70b-online",
        "perplexity/pplx-7b-chat",
        "perplexity/pplx-7b-online",
        "perplexity/sonar-medium-chat",
        "perplexity/sonar-medium-online",
        "perplexity/sonar-small-chat",
        "perplexity/sonar-small-online",
        "phind/phind-codellama-34b",
        "pygmalionai/mythalion-13b",
        "recursal/eagle-7b",
        "recursal/rwkv-5-3b-ai-town",
        "rwkv/rwkv-5-world-3b",
        "sao10k/fimbulvetr-11b-v2",
        "snowflake/snowflake-arctic-instruct",
        "sophosympatheia/midnight-rose-70b",
        "teknium/openhermes-2-mistral-7b",
        "teknium/openhermes-2.5-mistral-7b",
        "togethercomputer/stripedhyena-hessian-7b",
        "togethercomputer/stripedhyena-nous-7b",
        "undi95/remm-slerp-l2-13b",
        "undi95/remm-slerp-l2-13b:extended",
        "undi95/toppy-m-7b",
        "undi95/toppy-m-7b:free",
        "undi95/toppy-m-7b:nitro",
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
        "firellava-13b",  # https://openrouter.ai/models/fireworks/firellava-13b
        "gemini-pro-vision",
        "gemini-pro-1.5",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
        "llava-13b",  # https://openrouter.ai/models/haotian-liu/llava-13b
        "nous-hermes-2-vision-7b",  # https://openrouter.ai/models/nousresearch/nous-hermes-2-vision-7b
    ]
    META = RemoteLanguageModelMetaInfo(
        api_url="https://openrouter.ai/api/v1/chat/completions",
        required_config_fields=["api_key"],
        language_models=set(_SUPPORTED_MODELS) - set(_VISUAL_MODELS),
        visual_language_models=_VISUAL_MODELS,
        tool_models=_SUPPORTED_MODELS,
        online_models=[
            "pplx-70b-online",
            "pplx-7b-online",
            "sonar-medium-online",
            "sonar-small-online",
        ],
    )
    REQUEST_BODY_CLS = OpenRouterRequestBody
    RESPONSE_BODY_CLS = OpenRouterResponseBody
