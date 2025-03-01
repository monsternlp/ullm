import base64
import hashlib
import hmac
import json
import sys
from datetime import datetime
from time import time
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    conlist,
    validate_call,
)

from .base import (
    AssistantMessage,
    ChatMessage,
    FunctionObject,
    GenerateConfig,
    GenerationResult,
    HttpServiceModel,
    ImagePart,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    TextPart,
    Tool,
    ToolCall,
    ToolChoice,
    ToolMessage,
    UserMessage,
)
from .openai import OpenAICompatibleModel


class TencentImageURL(BaseModel):
    url: str = Field(..., serialization_alias="Url")


class TencentContent(BaseModel):
    type: Literal["text", "image_url"] = Field(..., serialization_alias="Type")
    text: Optional[str] = Field(None, serialization_alias="Text")
    image_url: Optional[TencentImageURL] = Field(None, serialization_alias="ImageUrl")


class TencentFunctionCall(BaseModel):
    name: str = Field(..., alias="Name")
    arguments: str = Field(..., alias="Arguments")

    model_config = ConfigDict(populate_by_name=True)


class TencentToolCall(BaseModel):
    id: str = Field(..., alias="Id")
    type: str = Field(..., alias="Type")
    function: TencentFunctionCall = Field(..., alias="Function")

    model_config = ConfigDict(populate_by_name=True)

    def to_standard(self):
        return ToolCall(
            id=self.id,
            type=self.type,
            function={"name": self.function.name, "arguments": json.loads(self.function.arguments)},
        )


class TencentChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"] = Field(..., serialization_alias="Role")
    content: Optional[str] = Field(None, serialization_alias="Content")
    contents: Optional[List[TencentContent]] = Field(None, serialization_alias="Contents")
    tool_call_id: Optional[str] = Field(None, serialization_alias="ToolCallId")
    tool_calls: Optional[List[TencentToolCall]] = Field(None, serialization_alias="ToolCalls")

    @classmethod
    @validate_call
    def from_standard(cls, message: ChatMessage):
        role, content, contents, tool_calls, tool_call_id = None, "", [], [], None
        if isinstance(message, UserMessage):
            role, has_image = "user", False
            for part in message.content:
                if isinstance(part, TextPart):
                    contents.append(part.model_dump())
                elif isinstance(part, ImagePart):
                    has_image = True
                    if part.data:
                        base64_data = base64.b64encode(part.data).decode("utf-8")
                        contents.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{part.mime_type};base64,{base64_data}"},
                            }
                        )
                    else:
                        contents.append({"type": "image_url", "image_url": {"url": part.url}})

            if not has_image:
                content = "\n".join([part["text"] for part in contents])
                contents = []

        elif isinstance(message, AssistantMessage):
            role = "assistant"
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": json.dumps(
                                    tool_call.function.arguments, ensure_ascii=False
                                ),
                            },
                        }
                    )

                tool_names = [tool_call.function.name for tool_call in message.tool_calls]
                content = f'Should call function(s): {",".join(tool_names)}'
            else:
                content = message.content

        elif isinstance(message, ToolMessage):
            role = "tool"
            content = message.tool_result
            tool_call_id = message.tool_call_id

        return cls(
            role=role,
            content=content,
            contents=contents or None,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls or None,
        )


class TencentFunctionObject(BaseModel):
    name: str = Field(..., serialization_alias="Name")
    description: Optional[str] = Field(..., serialization_alias="Description")
    parameters: str = Field(..., serialization_alias="Parameters")

    @classmethod
    def from_standard(cls, function: FunctionObject):
        parameters, required = {}, []
        for argument in function.arguments or []:
            if argument.required:
                required.append(argument.name)

            parameters[argument.name] = {
                "type": argument.type,
                "description": argument.description,
            }

        return cls(
            name=function.name,
            description=function.description,
            parameters=json.dumps(
                {"type": "object", "properties": parameters, "required": required},
                ensure_ascii=False,
            ),
        )


class TencentTool(BaseModel):
    type: Literal["function"] = Field(..., serialization_alias="Type")
    function: TencentFunctionObject = Field(..., serialization_alias="Function")

    @classmethod
    def from_standard(cls, tool: Tool):
        return cls(type=tool.type, function=TencentFunctionObject.from_standard(tool.function))


class TencentRequestBody(BaseModel):
    model: str = Field(..., serialization_alias="Model")
    messages: List[TencentChatMessage] = Field(..., serialization_alias="Messages")
    stream: Optional[bool] = Field(None, serialization_alias="Stream")
    stream_moderation: Optional[bool] = Field(None, serialization_alias="StreamModeration")
    top_p: Optional[float] = Field(None, serialization_alias="TopP")
    temperature: Optional[float] = Field(None, serialization_alias="Temperature")
    enable_enhancement: Optional[bool] = Field(None, serialization_alias="EnableEnhancement")
    tools: Optional[List[TencentTool]] = Field(None, serialization_alias="Tools")
    tool_choice: Optional[Literal["none", "auto", "custom"]] = Field(
        None, serialization_alias="ToolChoice"
    )
    custom_tool: Optional[TencentTool] = Field(None, serialization_alias="CustomTool")
    search_info: Optional[bool] = Field(None, serialization_alias="SearchInfo")
    citation: Optional[bool] = Field(None, serialization_alias="Citation")
    enable_speed_search: Optional[bool] = Field(None, serialization_alias="EnableSpeedSearch")
    enable_multi_media: Optional[bool] = Field(None, serialization_alias="EnableMultimedia")
    enable_deep_search: Optional[bool] = Field(None, serialization_alias="EnableDeepSearch")
    seed: Optional[int] = Field(None, serialization_alias="Seed")


class TencentUsage(BaseModel):
    prompt_tokens: int = Field(..., alias="PromptTokens")
    completion_tokens: int = Field(..., alias="CompletionTokens")
    total_tokens: int = Field(..., alias="TotalTokens")


class TencentResponseMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"] = Field(..., alias="Role")
    content: Optional[str] = Field(None, alias="Content")
    tool_call_id: Optional[str] = Field(None, alias="ToolCallId")
    tool_calls: Optional[List[TencentToolCall]] = Field(None, alias="ToolCalls")


class TencentResponseChoice(BaseModel):
    index: int = Field(..., alias="Index")
    finish_reason: str = Field(..., alias="FinishReason")
    message: TencentResponseMessage = Field(..., alias="Message")


class TencentResponseBodyData(BaseModel):
    id: str = Field(..., alias="Id")
    request_id: str = Field(..., alias="RequestId")
    created: int = Field(..., alias="Created")
    usage: TencentUsage = Field(..., alias="Usage")
    note: str = Field(..., alias="Note")
    choices: List[TencentResponseChoice] = Field(..., alias="Choices")
    error_message: Optional[dict] = Field(None, alias="ErrorMsg")
    moderation_level: Optional[str] = Field(None, alias="ModerationLevel")
    search_info: Optional[dict] = Field(None, alias="SearchInfo")
    replaces: Optional[List[dict]] = Field(None, alias="Replaces")


class TencentResponseBody(BaseModel):
    response: TencentResponseBodyData = Field(..., alias="Response")

    def to_standard(self, model: str = None):
        response = self.response
        tool_calls = None
        if response.choices[0].message.tool_calls:
            tool_calls = []
            for tool_call in response.choices[0].message.tool_calls:
                tool_calls.append(tool_call.to_standard())

        return GenerationResult(
            model=model or response.model,
            stop_reason=response.choices[0].finish_reason,
            content=response.choices[0].message.content,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )


@RemoteLanguageModel.register("tencent")
class TencentModel(HttpServiceModel):
    # reference: https://cloud.tencent.com/document/product/1729/105701
    META = RemoteLanguageModelMetaInfo(
        api_url="https://hunyuan.tencentcloudapi.com",
        language_models=[
            "hunyuan-lite",
            "hunyuan-standard",
            "hunyuan-standard-256K",
            "hunyuan-code",
            "hunyuan-role",
            "hunyuan-functioncall",
            "hunyuan-turbo",
            "hunyuan-turbo-latest",
            "hunyuan-turbo-20241223",
            "hunyuan-turbo-20241120",
            "hunyuan-turbos-20250226",
            "hunyuan-turbos-latestxrep",
            "hunyuan-large",
            "hunyuan-large-longcontext",
            "hunyuan-standard-online",
            "hunyuan-standard-256K-online",
            "hunyuan-code-online",
            "hunyuan-role-online",
            "hunyuan-functioncall-online",
            "hunyuan-turbo-online",
            "hunyuan-turbo-latest-online",
            "hunyuan-turbo-20241223-online",
            "hunyuan-turbo-20241120-online",
            "hunyuan-turbos-20250226-online",
            "hunyuan-turbos-latestxrep-online",
            "hunyuan-large-online",
            "hunyuan-large-longcontext-online",
        ],
        visual_language_models=[
            "hunyuan-vision",
            "hunyuan-lite-vision",
            "hunyuan-standard-vision",
            "hunyuan-turbo-vision",
        ],
        tool_models=[
            "hunyuan-turbo",
            "hunyuan-functioncall",
            "hunyuan-turbo-online",
            "hunyuan-functioncall-online",
        ],
        online_models=[
            "hunyuan-large-longcontext",
            "hunyuan-standard-online",
            "hunyuan-standard-256K-online",
            "hunyuan-code-online",
            "hunyuan-role-online",
            "hunyuan-functioncall-online",
            "hunyuan-turbo-online",
            "hunyuan-turbo-latest-online",
            "hunyuan-turbo-20241223-online",
            "hunyuan-turbo-20241120-online",
            "hunyuan-turbos-20250226-online",
            "hunyuan-turbos-latestxrep-online",
            "hunyuan-large-online",
            "hunyuan-large-longcontext-online",
        ],
        required_config_fields=["api_key", "secret_key", "region"],
    )
    REQUEST_BODY_CLS = TencentRequestBody
    RESPONSE_BODY_CLS = TencentResponseBody

    def _make_api_headers(self):
        # https://cloud.tencent.com/document/api/1729/101842
        return {
            "Host": "hunyuan.tencentcloudapi.com",
            "X-TC-Action": "ChatCompletions",
            "X-TC-VERSION": "2023-09-01",
            "X-TC-Timestamp": str(int(time())),
            "X-TC-Region": self.config.region,
            "Content-Type": "application/json",
        }

    @classmethod
    @validate_call
    def _convert_message(cls, message: ChatMessage) -> TencentChatMessage:
        return TencentChatMessage.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = [self._convert_message(msg) for msg in messages]
        if system:
            messages = [TencentChatMessage(role="system", content=system)] + messages

        return {"messages": messages}

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [TencentTool.from_standard(tool) for tool in tools]

        tencent_tool_choice, custom_tool = None, None
        if tools and tool_choice is not None:
            tencent_tool_choice = tool_choice.mode
            if tencent_tool_choice == "any":
                if len(tool_choice.functions) == 1:
                    tencent_tool_choice = "custom"
                    for tool in tools:
                        if tool.name == tool_choice.functions[0]:
                            custom_tool = tool
                            break

                    if not custom_tool:
                        tencent_tool_choice = "auto"
                else:
                    tencent_tool_choice = "auto"

        return {"tools": tools, "tool_choice": tencent_tool_choice, "custom_tool": custom_tool}

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        generation_config = {
            "model": self.model.replace("-online", ""),
            "top_p": config.top_p or self.config.top_p,
            "temperature": config.temperature or self.config.temperature,
        }
        if self.is_online_model():
            generation_config["enable_enhancement"] = True
        else:
            generation_config["enable_enhancement"] = False

        return generation_config

    def _sign(self, header, params):
        timestamp = int(header["X-TC-Timestamp"])
        data = json.dumps(params)
        service = "hunyuan"
        date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
        signature = self._get_tc3_signature(params, header, data, date)

        secret_id = self.config.api_key.get_secret_value()
        auth = (
            "TC3-HMAC-SHA256 "
            f"Credential={secret_id}/{date}/{service}/tc3_request, "
            "SignedHeaders=content-type;host, "
            f"Signature={signature}"
        )
        return auth

    def _get_tc3_signature(self, params, header, data, date):
        canonical_querystring = ""
        payload = data
        service = "hunyuan"

        if sys.version_info[0] == 3 and isinstance(payload, type("")):
            payload = payload.encode("utf8")

        payload_hash = hashlib.sha256(payload).hexdigest()

        canonical_headers = "content-type:%s\nhost:%s\n" % (header["Content-Type"], header["Host"])
        signed_headers = "content-type;host"
        canonical_request = "%s\n%s\n%s\n%s\n%s\n%s" % (
            "POST",
            "/",
            canonical_querystring,
            canonical_headers,
            signed_headers,
            payload_hash,
        )

        algorithm = "TC3-HMAC-SHA256"
        credential_scope = date + "/" + service + "/tc3_request"
        if sys.version_info[0] == 3:
            canonical_request = canonical_request.encode("utf8")

        digest = hashlib.sha256(canonical_request).hexdigest()
        str2sign = "%s\n%s\n%s\n%s" % (
            algorithm,
            header["X-TC-Timestamp"],
            credential_scope,
            digest,
        )

        def _hmac_sha256(key, msg):
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256)

        def _get_signature_key(key, date, service):
            k_date = _hmac_sha256(("TC3" + key).encode("utf-8"), date)
            k_service = _hmac_sha256(k_date.digest(), service)
            k_signing = _hmac_sha256(k_service.digest(), "tc3_request")
            return k_signing.digest()

        secret_key = self.config.secret_key.get_secret_value()
        signing_key = _get_signature_key(secret_key, date, service)
        signature = _hmac_sha256(signing_key, str2sign).hexdigest()
        return signature

    @validate_call
    def chat(
        self,
        messages: conlist(ChatMessage, min_length=1),
        config: Optional[GenerateConfig] = None,
        system: Optional[str] = None,
    ) -> GenerationResult:
        config = config or GenerateConfig()
        self._validate_model(messages)
        api_url = self._get_api_url()
        request_data = {
            "headers": self._make_api_headers(),
            "body": self._make_api_body(messages, config, system=system),
        }
        request_data["headers"]["Authorization"] = self._sign(
            request_data["headers"], request_data["body"]
        )
        return self._call_api(api_url, request_data)

    def _is_valid_response(cls, http_response):
        if http_response.status_code != 200:
            return False

        data = http_response.json()
        if "Response" not in data or data["Response"].get("Error"):
            return False

        return True


@RemoteLanguageModel.register("tencent-lke")
class TencentLKEModel(OpenAICompatibleModel):
    # reference: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Fm2vrveyu
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.lkeap.cloud.tencent.com/v1",
        language_models=[
            "deepseek-v3",
            "deepseek-r1",
        ],
        visual_language_models=[],
        tool_models=[],
        required_config_fields=["api_key"],
    )
