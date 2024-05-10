from enum import IntEnum
from hashlib import md5
from time import time
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, confloat, conint, conlist, validate_call

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


class SkyWorkChatMessage(BaseModel):
    role: Literal["system", "user", "bot"]
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

            return cls(role="bot", content=content)
        if isinstance(message, ToolMessage):
            content = (
                f"I called the function `{message.tool_name}` and "
                f"the response of that function is: {message.tool_result}"
            )
            return cls(role="user", content=content)


class SkyWorkRequestParams(BaseModel):
    # https://model-platform.tiangong.cn/api-reference
    generate_length: Optional[conint(ge=0, le=2048)] = None
    top_p: Optional[confloat(ge=0.1, le=1.0)] = None
    top_k: Optional[conint(ge=3, le=100)] = None
    repetition_penalty: Optional[confloat(ge=0.5, le=2.0)] = None
    length_penalty: Optional[confloat(ge=0.5, le=1.5)] = None
    min_len: Optional[conint(ge=1, le=10)] = None
    temperature: Optional[confloat(ge=0, le=1.0)] = None


class SkyWorkRequestBody(BaseModel):
    messages: conlist(SkyWorkChatMessage, min_length=1)
    model: str
    param: Optional[SkyWorkRequestParams] = None


class SkyWorkFinishReason(IntEnum):
    normal = 1
    length = 2


class SkyWorkStatus(IntEnum):
    generating = 1
    timeout = 2
    finished = 3
    content_filter = 4


class SkyWorkUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class SkyWorkResponseData(BaseModel):
    reply: str
    finish_reason: SkyWorkFinishReason
    status: SkyWorkStatus
    usage: SkyWorkUsage


class SkyWorkResponseBody(BaseModel):
    # https://model-platform.tiangong.cn/api-reference
    code: int
    code_msg: Optional[str] = None
    trace_id: Optional[str] = None
    resp_data: Optional[SkyWorkResponseData] = None

    def to_standard(self, model: str = None):
        finish_reason = "normal"
        status = self.resp_data.status
        if status == SkyWorkStatus.timeout:
            finish_reason = "timeout"
        elif status == SkyWorkStatus.content_filter:
            finish_reason = "content_filter"
        elif status == SkyWorkStatus.finished:
            finish_reason = self.resp_data.finish_reason.name
        else:
            finish_reason = status.name

        return GenerationResult(
            model=model,
            content=self.resp_data.reply,
            stop_reason=finish_reason,
            input_tokens=self.resp_data.usage.prompt_tokens,
            output_tokens=self.resp_data.usage.completion_tokens,
            total_tokens=self.resp_data.usage.total_tokens,
        )


@RemoteLanguageModel.register("skywork")
class SkyWorkModel(HttpServiceModel):
    META = RemoteLanguageModelMetaInfo(
        api_url="https://sky-api.singularity-ai.com/saas/api/v4/generate",
        language_models=["SkyChat-MegaVerse"],
        required_config_fields=["api_key", "secret_key"],
    )
    REQUEST_BODY_CLS = SkyWorkRequestBody
    RESPONSE_BODY_CLS = SkyWorkResponseBody

    def _make_api_headers(self):
        timestamp = int(time())
        sign = md5(
            (
                f"{self.config.api_key.get_secret_value()}"
                f"{self.config.secret_key.get_secret_value()}"
                f"{timestamp}"
            ).encode("utf-8")
        ).hexdigest()
        return {
            "app_key": self.config.api_key.get_secret_value(),
            "sign": sign,
            "timestamp": str(timestamp),
        }

    @classmethod
    @validate_call
    def _convert_message(cls, message: ChatMessage):
        return SkyWorkChatMessage.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = [self._convert_message(message) for message in messages]
        if system:
            messages = [SkyWorkChatMessage(role="system", content=system)] + messages

        return {"messages": messages}

    @validate_call
    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "param": {
                "generate_length": config.max_output_tokens or self.config.max_output_tokens,
                "top_p": config.top_p or self.config.top_p,
                "top_k": config.top_k or self.config.top_k,
                "temperature": config.temperature or self.config.temperature,
                "repetition_penalty": config.repetition_penalty,
            },
        }

    def _is_valid_response(cls, http_response):
        return http_response.status_code == 200 and http_response.json().get("code") == 200

    def _parse_error_response(self, http_response) -> GenerationResult:
        error_code, error_message = http_response.status_code, http_response.text
        response = http_response.json()
        if response.get("code") != 200:
            error_code = response["code"]

        return GenerationResult(
            model=self.model,
            finish_reason="error",
            content=f"Error {error_code}: {error_message}",
        )
