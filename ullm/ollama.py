import base64
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    PositiveInt,
    confloat,
    conint,
    conlist,
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


class OllamaChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    images: Optional[List[str]] = None

    @classmethod
    def from_standard(cls, message: ChatMessage):
        if isinstance(message, UserMessage):
            if isinstance(message.content, str):
                return cls(role="user", content=message.content)

            text_parts, image_parts = [], []
            for part in message.content:
                if isinstance(part, TextPart):
                    text_parts.append(part)
                elif part.data:
                    image_parts.append(part)

            content = "\n".join([part.text for part in text_parts])
            images = [
                base64.b64encode(image_part.data).decode("utf-8") for image_part in image_parts
            ]
            return cls(role="user", content=content, images=images)
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


class OllamaRequestOptions(BaseModel):
    # reference: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    mirostat: Optional[conint(ge=0, le=2)] = None
    mirostat_eta: Optional[NonNegativeFloat] = None
    mirostat_tau: Optional[NonNegativeFloat] = None
    num_ctx: Optional[PositiveInt] = None
    repeat_last_n: Optional[conint(ge=-1)] = None
    repeat_penalty: Optional[float] = None
    temperature: Optional[confloat(ge=0.0, le=2.0)] = None
    seed: Optional[int] = None
    # NOTE: stop is a list based on this - https://github.com/ollama/ollama/pull/442
    stop: Optional[List[str]] = None
    tfs_z: Optional[confloat(ge=1.0)] = None
    num_predict: Optional[PositiveInt] = None
    top_k: Optional[PositiveInt] = None
    top_p: Optional[confloat(ge=0.0, le=1.0)] = None


class OllamaRequestBody(BaseModel):
    # reference: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    model: str
    messages: conlist(OllamaChatMessage, min_length=1)
    options: Optional[OllamaRequestOptions] = None
    response_format: Optional[Literal["json"]] = Field(None, alias="format")
    stream: Optional[bool] = False
    keep_alive: Optional[str] = None


class OllamaResponseBody(BaseModel):
    model: str
    created_at: datetime
    message: OllamaChatMessage
    done: Optional[bool]
    load_duration: int
    eval_duration: int
    prompt_eval_duration: int
    total_duration: int
    eval_count: int

    def to_standard(self, model: str = None):
        return GenerationResult(
            model=model or self.model,
            stop_reason="stop",
            content=self.message.content,
            input_tokens=None,
            output_tokens=self.eval_count,
            total_tokens=None,
        )


@RemoteLanguageModel.register("ollama")
class OllamaModel(HttpServiceModel):
    META = RemoteLanguageModelMetaInfo(
        required_config_fields=["api_url", "is_visual_model", "is_tool_model"],
    )
    REQUEST_BODY_CLS = OllamaRequestBody
    RESPONSE_BODY_CLS = OllamaResponseBody

    def _make_api_headers(self):
        pass

    @classmethod
    def _convert_message(cls, message: ChatMessage):
        return OllamaChatMessage.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = [self._convert_message(message) for message in messages]
        if system:
            messages = [OllamaChatMessage(role="system", content=system)] + messages

        return {"messages": messages}

    @validate_call
    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "options": {
                "repeat_penalty": config.repetition_penalty,
                "num_ctx": config.max_input_tokens or self.config.max_input_tokens,
                "temperature": config.temperature or self.config.temperature,
                "stop": config.stop_sequences or self.config.stop_sequences,
                "num_predict": config.max_output_tokens or self.config.max_output_tokens,
                "top_k": config.top_k or self.config.top_k,
                "top_p": config.top_p or self.config.top_p,
            },
            "format": "json" if config.response_format == "json_object" else None,
        }
