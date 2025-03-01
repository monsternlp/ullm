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
    # reference: https://docs.together.ai/reference/chat-completions-1
    ## excluded parameters
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
    min_p: Optional[float] = None


@RemoteLanguageModel.register("together")
class TogetherAIModel(OpenAICompatibleModel):
    # reference: https://docs.together.ai/docs/quickstart
    ## text models: https://docs.together.ai/docs/chat-models
    _LANGUAGE_MODELS = [
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-V3",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "meta-llama/Meta-Llama-3-70B-Instruct-Lite",
        "meta-llama/Llama-3-8b-chat-hf",
        "meta-llama/Llama-3-70b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/QwQ-32B-Preview",
        "microsoft/WizardLM-2-8x22B",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "databricks/dbrx-instruct",
        "google/gemma-2b-it",
        "Gryphe/MythoMax-L2-13b",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "Qwen/Qwen2-72B-Instruct",
        "upstage/SOLAR-10.7B-Instruct-v1.0",
    ]
    ## vision models: https://docs.together.ai/docs/vision-models
    _VISION_MODELS = [
        "meta-llama/Llama-Vision-Free",
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "Qwen/Qwen2-VL-72B-Instruct",
    ]
    _FULL_SUPPORTED_MODELS = _LANGUAGE_MODELS + _VISION_MODELS
    _MODEL_MAPPINGS = {
        full_model_name.split("/")[1]: full_model_name for full_model_name in _FULL_SUPPORTED_MODELS
    }
    _SUPPORTED_MODELS = sorted(_MODEL_MAPPINGS.keys())
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.together.xyz/v1/chat/completions",
        # https://docs.together.ai/docs/inference-models#chat-models
        language_models=[full_model_name.split("/")[1] for full_model_name in _LANGUAGE_MODELS],
        visual_language_models=[
            full_model_name.split("/")[1] for full_model_name in _VISION_MODELS
        ],
        tool_models=[
            "Meta-Llama-3.1-8B-Instruct-Turbo",
            "Meta-Llama-3.1-70B-Instruct-Turbo",
            "Meta-Llama-3.1-405B-Instruct-Turbo",
            "Llama-3.3-70B-Instruct-Turbo",
            "Mixtral-8x7B-Instruct-v0.1",
            "Mistral-7B-Instruct-v0.1",
            "Qwen2.5-7B-Instruct-Turbo",
            "Qwen2.5-72B-Instruct-Turbo",
        ],
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
