from typing import Any, Dict, Literal, Optional

from pydantic import (
    BaseModel,
    NonNegativeFloat,
    PositiveInt,
    conlist,
    model_validator,
    validate_call,
)

from .base import (
    AssistantMessage,
    ChatMessage,
    GenerateConfig,
    GenerationResult,
    HttpServiceModel,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    TextPart,
    ToolMessage,
    UserMessage,
)


class CloudflareChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @classmethod
    def from_standard(cls, message: ChatMessage):
        if isinstance(message, UserMessage):
            content = message.content
            if isinstance(message.content, list):
                text_parts = []
                for part in message.content:
                    if isinstance(part, TextPart):
                        text_parts.append(part)
                    elif part.data:
                        pass

                content = "\n".join([part.text for part in text_parts])
            return cls(role="user", content=content)
        if isinstance(message, AssistantMessage):
            content = message.content
            if message.tool_calls:  # NOTE: API 文档中未定义相关行为
                content = ""
                for tool_call in message.tool_calls:
                    content += (
                        f"You should call the function `{tool_call.function.name}` with arguments: "
                        f"{tool_call.function.arguments}\n"
                    )

            return cls(role="assistant", content=content)
        if isinstance(message, ToolMessage):
            content = (
                f"I called the function `{message.tool_name}` and "
                f"the response of that function is: {message.tool_result}"
            )
            return cls(role="user", content=content)


class CloudflareRequestBody(BaseModel):
    # https://developers.cloudflare.com/workers-ai/models/deepseek-coder-6.7b-base-awq/
    lora: Optional[str] = None
    max_tokens: Optional[PositiveInt] = None
    prompt: Optional[str] = None
    messages: Optional[conlist(CloudflareChatMessage, min_length=1)] = None
    raw: Optional[bool] = None
    stream: Optional[bool] = None
    temperature: Optional[NonNegativeFloat] = None

    @model_validator(mode="before")
    def check_promt_or_messages(cls, data):
        assert data.get("prompt") or data.get("messages")
        return data


class CloudflareResponseResult(BaseModel):
    response: str


class CloudflareResponseBody(BaseModel):
    # https://developers.cloudflare.com/workers-ai/models/deepseek-coder-6.7b-base-awq/
    success: Optional[bool] = True
    result: CloudflareResponseResult

    def to_standard(self, model: str = None):
        return GenerationResult(
            model=model,
            stop_reason="stop",
            content=self.result.response,
        )


@RemoteLanguageModel.register("cloudflare")
class CloudflareModel(HttpServiceModel):
    # https://developers.cloudflare.com/workers-ai/models/
    _MODEL_MAPPINGS = {
        "deepseek-coder-6.7b-base-awq": "@hf/thebloke/deepseek-coder-6.7b-base-awq",
        "deepseek-coder-6.7b-instruct-awq": "@hf/thebloke/deepseek-coder-6.7b-instruct-awq",
        "deepseek-math-7b-instruct": "@cf/deepseek-ai/deepseek-math-7b-instruct",
        "deepseek-r1-distill-qwen-32b": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "discolm-german-7b-v1-awq": "@cf/thebloke/discolm-german-7b-v1-awq",
        "falcon-7b-instruct": "@cf/tiiuae/falcon-7b-instruct",
        "gemma-2b-it-lora": "@cf/google/gemma-2b-it-lora",
        "gemma-7b-it": "@hf/google/gemma-7b-it",
        "gemma-7b-it-lora": "@cf/google/gemma-7b-it-lora",
        "hermes-2-pro-mistral-7b": "@hf/nousresearch/hermes-2-pro-mistral-7b",
        "llama-2-13b-chat-awq": "@hf/thebloke/llama-2-13b-chat-awq",
        "llama-2-7b-chat-fp16": "@cf/meta/llama-2-7b-chat-fp16",
        "llama-2-7b-chat-hf-lora": "@cf/meta-llama/llama-2-7b-chat-hf-lora",
        "llama-3-8b-instruct-awq": "@cf/meta/llama-3-8b-instruct-awq",
        "llama-3-8b-instruct": "@cf/meta/llama-3-8b-instruct",
        "llama-3.1-8b-instruct-awq": "@cf/meta/llama-3.1-8b-instruct-awq",
        "llama-3.1-8b-instruct-fp8": "@cf/meta/llama-3.1-8b-instruct-fp8",
        "llama-3.1-8b-instruct": "@cf/meta/llama-3.1-8b-instruct",
        "llamaguard-7b-awq": "@hf/thebloke/llamaguard-7b-awq",
        "mistral-7b-instruct-v0.1": "@cf/mistral/mistral-7b-instruct-v0.1",
        "mistral-7b-instruct-v0.1-awq": "@hf/thebloke/mistral-7b-instruct-v0.1-awq",
        "mistral-7b-instruct-v0.2": "@hf/mistral/mistral-7b-instruct-v0.2",
        "mistral-7b-instruct-v0.2-lora": "@cf/mistral/mistral-7b-instruct-v0.2-lora",
        "neural-chat-7b-v3-1-awq": "@hf/thebloke/neural-chat-7b-v3-1-awq",
        "openchat-3.5-0106": "@cf/openchat/openchat-3.5-0106",
        "openhermes-2.5-mistral-7b-awq": "@hf/thebloke/openhermes-2.5-mistral-7b-awq",
        "phi-2": "@cf/microsoft/phi-2",
        "qwen1.5-0.5b-chat": "@cf/qwen/qwen1.5-0.5b-chat",
        "qwen1.5-1.8b-chat": "@cf/qwen/qwen1.5-1.8b-chat",
        "qwen1.5-7b-chat-awq": "@cf/qwen/qwen1.5-7b-chat-awq",
        "qwen1.5-14b-chat-awq": "@cf/qwen/qwen1.5-14b-chat-awq",
        "sqlcoder-7b-2": "@cf/defog/sqlcoder-7b-2",
        "starling-lm-7b-beta": "@hf/nexusflow/starling-lm-7b-beta",
        "tinyllama-1.1b-chat-v1.0": "@cf/tinyllama/tinyllama-1.1b-chat-v1.0",
        "una-cybertron-7b-v2-bf16": "@cf/fblgit/una-cybertron-7b-v2-bf16",
        "zephyr-7b-beta-awq": "@hf/thebloke/zephyr-7b-beta-awq",
    }
    _LANGUAGE_MODELS = list(_MODEL_MAPPINGS)
    META = RemoteLanguageModelMetaInfo(
        language_models=_LANGUAGE_MODELS,
        visual_language_models=[],
        tool_models=[
            "hermes-2-pro-mistral-7b",
        ],
        online_models=[],
        required_config_fields=["api_key", "cf_account_id"],
    )
    REQUEST_BODY_CLS = CloudflareRequestBody
    RESPONSE_BODY_CLS = CloudflareResponseBody
    _CF_WORKER_AI_URL_TEMPLATE = (
        "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    )

    def _make_api_headers(self):
        return {"Authorization": f"Bearer {self.config.api_key.get_secret_value()}"}

    def _get_api_url(self):
        return self._CF_WORKER_AI_URL_TEMPLATE.format(
            account_id=self.config.cf_account_id.get_secret_value(),
            model=self._MODEL_MAPPINGS[self.model],
        )

    def _convert_message(cls, message: ChatMessage):
        return CloudflareChatMessage.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = [self._convert_message(message) for message in messages]
        if system:
            messages = [CloudflareChatMessage(role="system", content=system)] + messages

        return {"messages": messages}

    @validate_call
    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "temperature": config.temperature or self.config.temperature,
        }
