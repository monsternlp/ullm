from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, PositiveInt, conint

from .base import (
    GenerateConfig,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import (
    OpenAICompatibleModel,
    OpenAIRequestBody,
)


class TogetherAIRequestBody(OpenAIRequestBody):
    # reference: https://docs.together.ai/reference/chat-completions
    ## excluded parameters
    logit_bias: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)

    ## different parameters
    logprobs: Optional[conint(ge=0, le=20)] = None
    stop: Optional[List[str]] = None
    response_format: Optional[Dict[Literal["type"], Literal["json_object"]]] = None

    # TogetherAI-specific parameters
    top_k: Optional[PositiveInt] = None
    repetition_penalty: Optional[float] = None
    echo: Optional[bool] = None
    safety_model: Optional[str] = None


@RemoteLanguageModel.register("together")
class TogetherAIModel(OpenAICompatibleModel):
    # reference: https://docs.together.ai/docs/quickstart
    _FULL_SUPPORTED_MODELS = [
        "Austism/chronos-hermes-13b",
        "Gryphe/MythoMax-L2-13b",
        "NousResearch/Nous-Capybara-7B-V1p9",
        "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
        "NousResearch/Nous-Hermes-2-Yi-34B",
        "NousResearch/Nous-Hermes-Llama2-13b",
        "NousResearch/Nous-Hermes-llama-2-7b",
        "Open-Orca/Mistral-7B-OpenOrca",
        "Qwen/Qwen1.5-0.5B-Chat",
        "Qwen/Qwen1.5-1.8B-Chat",
        "Qwen/Qwen1.5-110B-Chat",
        "Qwen/Qwen1.5-14B-Chat",
        "Qwen/Qwen1.5-32B-Chat",
        "Qwen/Qwen1.5-4B-Chat",
        "Qwen/Qwen1.5-72B-Chat",
        "Qwen/Qwen1.5-7B-Chat",
        "Snowflake/snowflake-arctic-instruct",
        "Undi95/ReMM-SLERP-L2-13B",
        "Undi95/Toppy-M-7B",
        "WizardLM/WizardLM-13B-V1.2",
        "allenai/OLMo-7B",
        "allenai/OLMo-7B-Instruct",
        "allenai/OLMo-7B-Twin-2T",
        "codellama/CodeLlama-13b-Instruct-hf",
        "codellama/CodeLlama-34b-Instruct-hf",
        "codellama/CodeLlama-70b-Instruct-hf",
        "codellama/CodeLlama-7b-Instruct-hf",
        "cognitivecomputations/dolphin-2.5-mixtral-8x7",
        "databricks/dbrx-instruct",
        "deepseek-ai/deepseek-coder-33b-instruct",
        "deepseek-ai/deepseek-llm-67b-chat",
        "garage-bAInd/Platypus2-70B-instruct",
        "google/gemma-2b-it",
        "google/gemma-7b-it",
        "lmsys/vicuna-13b-v1.5",
        "lmsys/vicuna-7b-v1.5",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3-70b-chat-hf",
        "meta-llama/Llama-3-8b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "openchat/openchat-3.5-1210",
        "snorkelai/Snorkel-Mistral-PairRM-DPO",
        "teknium/OpenHermes-2-Mistral-7B",
        "teknium/OpenHermes-2p5-Mistral-7B",
        "togethercomputer/Llama-2-7B-32K-Instruct",
        "togethercomputer/RedPajama-INCITE-7B-Chat",
        "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
        "togethercomputer/StripedHyena-Nous-7B",
        "togethercomputer/alpaca-7b",
        "upstage/SOLAR-10.7B-Instruct-v1.0",
        "zero-one-ai/Yi-34B-Chat",
    ]
    _MODEL_MAPPINGS = {
        full_model_name.split("/")[1]: full_model_name for full_model_name in _FULL_SUPPORTED_MODELS
    }
    _SUPPORTED_MODELS = sorted(_MODEL_MAPPINGS.keys())
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.together.xyz/v1/chat/completions",
        # https://docs.together.ai/docs/inference-models#chat-models
        language_models=_SUPPORTED_MODELS,
        visual_language_models=[],
        tool_models=_SUPPORTED_MODELS,
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = TogetherAIRequestBody

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        response_format = None
        if config.response_format == "json_object":
            response_format = {"type": "json_object"}

        return {
            "model": self._MODEL_MAPPINGS[self.model],
            "frequency_penalty": config.frequency_penalty,
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "presence_penalty": config.presence_penalty,
            "response_format": response_format,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
            "top_k": config.top_k or self.config.top_k,
            "repetition_penalty": config.repetition_penalty,
        }
