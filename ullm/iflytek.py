import base64
import hashlib
import hmac
import json
from contextlib import closing
from typing import List, Literal, Optional
from urllib.parse import urlencode, urlparse, urlunparse

import arrow
import websocket
from pydantic import (
    BaseModel,
    NonNegativeInt,
    confloat,
    conint,
    conlist,
    validate_call,
)

from .base import (
    AssistantMessage,
    ChatMessage,
    FunctionCall,
    GenerateConfig,
    GenerationResult,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from .openai import OpenAIFunctionObject


class IflyTekHeader(BaseModel):
    app_id: str
    uid: Optional[str] = None


class IflyTekChatParameters(BaseModel):
    domain: Literal["general", "generalv2", "generalv3", "generalv3.5"]
    temperature: Optional[confloat(gt=0.0, le=1.0)] = None
    max_tokens: Optional[int] = None
    top_k: Optional[conint(ge=1, le=6)] = None
    chat_id: Optional[str] = None


class IflyTekParameters(BaseModel):
    chat: IflyTekChatParameters


class IflyTekChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    function_call: Optional[FunctionCall] = None

    @classmethod
    def from_standard(cls, message: ChatMessage):
        role, content, function_call = message.role, "", None
        if isinstance(message, UserMessage):
            for part in message.content:
                if isinstance(part, TextPart):
                    content += part.text + "\n"
                else:
                    pass
        elif isinstance(message, AssistantMessage):
            content = message.content
            if message.tool_calls:  # NOTE: API 文档中未定义相关行为
                content = ""
                for tool_call in message.tool_calls:
                    content += (
                        f"You should call the function `{tool_call.function.name}` with arguments: "
                        f"{tool_call.function.arguments}\n"
                    )

                if len(message.tool_calls) == 1:
                    function_call = message.tool_calls[0].function

        elif isinstance(message, ToolMessage):
            # NOTE: API 文档中未定义相关行为
            role = "user"
            content = (
                f"I called the function `{message.tool_name}` and "
                f"the response of that function is: {message.tool_result}"
            )

        return cls(role=role, content=content.strip(), function_call=function_call)


class IflyTekPayloadMessage(BaseModel):
    text: List[IflyTekChatMessage]


class IflyTekPayloadFunctions(BaseModel):
    text: List[OpenAIFunctionObject]


class IflyTekPayload(BaseModel):
    message: IflyTekPayloadMessage
    functions: Optional[IflyTekPayloadFunctions] = None


class IflyTekRequestBody(BaseModel):
    # https://www.xfyun.cn/doc/spark/Web.html#_1-%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E
    header: IflyTekHeader
    parameter: IflyTekParameters
    payload: IflyTekPayload


class IflyTekResponseHeader(BaseModel):
    code: int
    message: str
    sid: str
    status: conint(ge=0, le=2)


class IflyTekResponsePayloadChoices(BaseModel):
    status: conint(ge=0, le=2)
    seq: NonNegativeInt
    text: List[IflyTekChatMessage]


class _IflyTekResponsePayloadUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class IflyTekResponsePayloadUsage(BaseModel):
    text: _IflyTekResponsePayloadUsage


class IflyTekResponsePayload(BaseModel):
    choices: IflyTekResponsePayloadChoices
    usage: Optional[IflyTekResponsePayloadUsage] = None


class IflyTekResponse(BaseModel):
    # https://www.xfyun.cn/doc/spark/Web.html#_1-%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E
    header: IflyTekResponseHeader
    payload: Optional[IflyTekResponsePayload] = None

    def to_standard(self, model: str):
        message = self.payload.choices.text[0]
        tool_calls = None
        if message.function_call:
            tool_calls = [ToolCall(function=message.function_call)]

        return GenerationResult(
            model=model,
            stop_reason="stop",
            content=message.content,
            tool_calls=tool_calls,
            input_tokens=self.payload.usage.text.prompt_tokens,
            output_tokens=self.payload.usage.text.completion_tokens,
            total_tokens=self.payload.usage.text.total_tokens,
        )


@RemoteLanguageModel.register("iflytek")
class IflyTekModel(RemoteLanguageModel):
    # Reference: https://www.xfyun.cn/doc/spark/Web.html#_1-%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E
    # TODO: 据说会升级 API 为 OpenAI 的格式: https://github.com/iflytek/spark-ai-python/issues/17
    META = RemoteLanguageModelMetaInfo(
        model_api_url_mappings={
            "SparkDesk-v3.5": "wss://spark-api.xf-yun.com/v3.5/chat",
            "SparkDesk-v3.1": "wss://spark-api.xf-yun.com/v3.1/chat",
            "SparkDesk-v2": "wss://spark-api.xf-yun.com/v2.1/chat",
            "SparkDesk-v1": "wss://spark-api.xf-yun.com/v1.1/chat",
        },
        language_models=["SparkDesk-v3.5", "SparkDesk-v3.1", "SparkDesk-v2", "SparkDesk-v1"],
        # https://www.xfyun.cn/doc/spark/Web.html#_2-function-call%E8%AF%B4%E6%98%8E
        tool_models=["SparkDesk-v3.1", "SparkDesk-v3.5"],
        required_config_fields=["app_id", "api_key", "secret_key"],
    )
    MODEL_TO_DOMAIN = {
        "SparkDesk-v3.5": "generalv3.5",
        "SparkDesk-v3.1": "generalv3",
        "SparkDesk-v2": "generalv2",
        "SparkDesk-v1": "generalv1",
    }

    def _get_api_url(self):
        # generate timestamp by RFC1123
        timestamp = arrow.utcnow().format(arrow.FORMAT_RFC1123)

        # urlparse
        api_url = str(self.META.model_api_url_mappings[self.model])
        parsed_url = urlparse(api_url)
        host = parsed_url.netloc
        path = parsed_url.path

        signature_origin = f"host: {host}\ndate: {timestamp}\nGET {path} HTTP/1.1"

        # encrypt using hmac-sha256
        signature_sha = hmac.new(
            self.config.secret_key.get_secret_value().encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding="utf-8")
        authorization_origin = (
            f'api_key="{self.config.api_key.get_secret_value()}", '
            f'algorithm="hmac-sha256", '
            f'headers="host date request-line", '
            f'signature="{signature_sha_base64}"'
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            encoding="utf-8"
        )

        # generate url
        params_dict = {"authorization": authorization, "date": timestamp, "host": host}
        encoded_params = urlencode(params_dict)
        url = urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                encoded_params,
                parsed_url.fragment,
            )
        )
        return url

    @validate_call
    def chat(
        self,
        messages: conlist(ChatMessage, min_length=1),
        config: Optional[GenerateConfig] = None,
        system: Optional[str] = None,
    ) -> GenerationResult:
        self._validate_model(messages)
        api_url = self._get_api_url()

        messages = [IflyTekChatMessage.from_standard(message) for message in messages]
        if system:
            messages = [IflyTekChatMessage(role="system", content=system)] + messages

        config = config or GenerateConfig()
        data = IflyTekRequestBody.model_validate(
            {
                "header": {"app_id": self.config.app_id},
                "parameter": {
                    "chat": {
                        "domain": self.MODEL_TO_DOMAIN[self.model],
                        "temperature": config.temperature or self.config.temperature,
                        "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
                        "top_k": config.top_k or self.config.top_k,
                    }
                },
                "payload": {"message": {"text": messages}},
            }
        )
        if self.is_tool_model() and config.tools:
            data.payload.functions = IflyTekPayloadFunctions(
                text=[OpenAIFunctionObject.from_standard(tool.function) for tool in config.tools]
            )

        result, original_responses = None, []
        with closing(websocket.create_connection(api_url)) as conn:
            conn.send(json.dumps(data.model_dump(exclude_none=True)))
            status, code = -1, 0
            while status != 2 and code == 0:
                message = conn.recv()
                data = json.loads(message)
                original_responses.append(data)
                try:
                    data = IflyTekResponse.model_validate(data)
                    code = data.header.code
                    if code != 0:
                        result = GenerationResult(
                            model=self.model,
                            stop_reason="error",
                            content=f"Error {code}: {data.header.message}",
                        )
                        break

                    status = data.header.status
                except Exception:
                    result = GenerationResult(
                        model=self.model, stop_reason="error", content=f"Bad response data: {data}"
                    )
                    break

        if not result:
            original_responses = sorted(
                original_responses, key=lambda x: x["payload"]["choices"]["seq"]
            )
            message, function_call = IflyTekChatMessage(role="assistant", content=""), None
            for response in original_responses:
                message.content += response["payload"]["choices"]["text"][0]["content"] or ""
                if response["payload"]["choices"]["text"][0].get("function_call"):
                    function_call = response["payload"]["choices"]["text"][0]["function_call"]

            message.function_call = function_call
            final_response = IflyTekResponse(
                header=original_responses[-1]["header"],
                payload={
                    "choices": {
                        "status": original_responses[-1]["payload"]["choices"]["status"],
                        "seq": original_responses[-1]["payload"]["choices"]["seq"],
                        "text": [message],
                    },
                    "usage": original_responses[-1]["payload"]["usage"],
                },
            )
            result = GenerationResult(
                model=self.model,
                stop_reason="stop",
                content=final_response.payload.choices.text[0].content,
                tool_calls=None
                if not message.function_call
                else [ToolCall(function=message.function_call)],
                input_tokens=final_response.payload.usage.text.prompt_tokens,
                output_tokens=final_response.payload.usage.text.completion_tokens,
                total_tokens=final_response.payload.usage.text.total_tokens,
            )

        result.original_result = json.dumps(original_responses, ensure_ascii=False)
        return result
