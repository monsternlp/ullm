import base64
import json
import time
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, validate_call

from .base import (
    AssistantMessage,
    ChatMessage,
    FunctionObject,
    GenerateConfig,
    GenerationResult,
    HttpServiceModel,
    ImagePart,
    JsonSchemaObject,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    TextPart,
    Tool,
    ToolCall,
    ToolChoice,
    ToolMessage,
    UserMessage,
)


class OpenAISystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str

    def to_standard(self) -> str:
        return self.content


class OpenAIImageURL(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = Field(default="auto")


class OpenAITextPart(TextPart):
    @classmethod
    def from_standard(cls, text_part: TextPart):
        return cls(text=text_part.text)

    def to_standard(self) -> TextPart:
        return self


class OpenAIImagePart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: OpenAIImageURL

    @classmethod
    def from_standard(cls, image_part: ImagePart):
        if image_part.url:
            return cls(image_url=OpenAIImageURL(url=str(image_part.url)))

        else:
            base64_data = base64.b64encode(image_part.data).decode("utf-8")
            return cls(
                image_url=OpenAIImageURL(url=f"data:{image_part.mime_type};base64,{base64_data}")
            )

    def to_standard(self) -> ImagePart:
        if self.image_url.url.startswith("data:"):
            # data:[<MIME-type>][;charset=<encoding>][;base64],<data>
            _mime_type_str, _base64_data_str = self.image_url.url.split(",", 1)
            mime_type = _mime_type_str.split(":")[1].split(";")[0]
            data = base64.b64decode(_base64_data_str)
            return ImagePart(data=data, mime_type=mime_type)
        else:
            return ImagePart(url=HttpUrl(self.image_url.url))


class OpenAIUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: Union[
        str, Annotated[List[Union[OpenAITextPart, OpenAIImagePart]], Field(min_length=1)]
    ]

    @classmethod
    def from_standard(cls, user_message: UserMessage):
        if all(isinstance(part, TextPart) for part in user_message.content):
            content = "\n".join([part.text for part in user_message.content])  # type: ignore
            return cls(content=content)

        parts = []
        for part in user_message.content:
            if isinstance(part, TextPart):
                parts.append(OpenAITextPart.from_standard(part))
            elif isinstance(part, ImagePart):
                parts.append(OpenAIImagePart.from_standard(part))

        return cls(content=parts)

    def to_standard(self) -> UserMessage:
        if isinstance(self.content, str):
            return UserMessage(content=[TextPart(text=self.content)])
        else:
            parts = []
            for part in self.content:
                parts.append(part.to_standard())
            return UserMessage(content=parts)


class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str


class OpenAIToolCall(ToolCall):
    function: Optional[OpenAIFunctionCall] = None

    @classmethod
    def from_standard(cls, tool_call):
        return cls(
            id=tool_call.id,
            type=tool_call.type,
            function={
                "name": tool_call.function.name,
                "arguments": json.dumps(tool_call.function.arguments, ensure_ascii=False),
            },
        )

    def to_standard(self):
        if not self.function:
            raise ValueError("Function call is required for tool call conversion")

        return ToolCall(
            id=self.id,
            type=self.type,
            function={
                "name": self.function.name,
                "arguments": json.loads(self.function.arguments),
            },
        )


class OpenAIAssistantMessage(AssistantMessage):
    tool_calls: Optional[List[OpenAIToolCall]] = None
    reasoning_content: Optional[str] = None

    @classmethod
    def from_standard(cls, message: AssistantMessage):
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                tool_calls.append(OpenAIToolCall.from_standard(tool_call))

        return cls(
            role=message.role,
            content=message.content,
            tool_calls=tool_calls,
        )

    def to_standard(self) -> AssistantMessage:
        tool_calls = None
        if self.tool_calls:
            tool_calls = [tool_call.to_standard() for tool_call in self.tool_calls]
        return AssistantMessage(
            content=self.content, tool_calls=tool_calls, reasoning_content=self.reasoning_content
        )


class OpenAIToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: Optional[str] = Field(default="success")
    tool_call_id: Optional[str] = Field(default_factory=lambda: uuid4().hex)

    @classmethod
    def from_standard(cls, tool_message: ToolMessage):
        return cls(
            role=tool_message.role,
            content=tool_message.tool_result,
            tool_call_id=tool_message.tool_call_id,
        )

    def to_standard(self) -> ToolMessage:
        return ToolMessage(
            tool_call_id=self.tool_call_id, tool_name="", tool_result=self.content or ""
        )


OpenAIChatMessage = Union[
    OpenAISystemMessage, OpenAIUserMessage, OpenAIToolMessage, OpenAIAssistantMessage
]


class OpenAIFunctionObject(BaseModel):
    name: str
    description: Optional[str] = Field(default=None)
    parameters: Optional[JsonSchemaObject] = Field(default=None)

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
            parameters={"type": "object", "properties": parameters, "required": required},
        )

    def to_standard(self) -> FunctionObject:
        arguments = []
        if self.parameters and self.parameters.properties:
            for name, prop in self.parameters.properties.items():
                arguments.append(
                    {
                        "name": name,
                        "type": prop.get("type", "string"),
                        "description": prop.get("description", ""),
                        "required": name in (self.parameters.required or []),  # type: ignore
                    }
                )
        return FunctionObject(
            name=self.name, description=self.description or "", arguments=arguments
        )


class OpenAITool(BaseModel):
    type: Literal["function"]
    function: OpenAIFunctionObject

    @classmethod
    def from_standard(cls, tool: Tool):
        return cls(type=tool.type, function=OpenAIFunctionObject.from_standard(tool.function))

    def to_standard(self) -> Tool:
        return Tool(type=self.type, function=self.function.to_standard())


class OpenAIToolChoice(BaseModel):
    type: Literal["function"]
    function: Dict[Literal["name"], str]

    def to_standard(self) -> ToolChoice:
        return ToolChoice(mode="any", functions=[self.function["name"]])


class OpenAIRequestBody(BaseModel):
    # https://platform.openai.com/docs/api-reference/chat/create
    # NOTE:
    # 1. gpt-4-vision 不能设置 logprobs/logit_bias/tools/tool_choice/response_format 几个参数
    # 2. 只有 gpt-4-turbo 系列模型和比 gpt-3.5-turbo-1106 更新的模型可以使用 response_format 参数
    messages: Annotated[List[OpenAIChatMessage], Field(min_length=1)]
    model: str
    frequency_penalty: Optional[Annotated[float, Field(ge=-2.0, le=2.0)]] = Field(default=None)
    logit_bias: Optional[Dict[str, int]] = Field(default=None)
    logprobs: Optional[bool] = Field(default=None)
    top_logprobs: Optional[Annotated[int, Field(ge=0, le=20)]] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    n: Optional[Annotated[int, Field(ge=1, le=128)]] = Field(default=1)
    presence_penalty: Optional[Annotated[float, Field(ge=-2.0, le=2.0)]] = Field(default=None)
    response_format: Optional[Dict[Literal["type"], Literal["text", "json_object"]]] = Field(
        default=None
    )
    seed: Optional[int] = Field(default=None)
    stop: Optional[Union[str, List[str]]] = Field(default=None)
    stream: Optional[bool] = Field(default=False)
    temperature: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = Field(default=None)
    top_p: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(default=None)
    tools: Optional[List[OpenAITool]] = Field(default=None)
    tool_choice: Optional[Union[Literal["auto", "none"], OpenAIToolChoice]] = Field(default=None)
    user: Optional[str] = Field(default=None)

    def to_standard(self) -> Dict[str, Any]:
        standard_messages: List[ChatMessage] = []
        system_message: Optional[str] = None

        for msg in self.messages:
            if isinstance(msg, OpenAISystemMessage):
                system_message = msg.to_standard()
            elif isinstance(msg, (OpenAIUserMessage, OpenAIAssistantMessage, OpenAIToolMessage)):
                standard_messages.append(msg.to_standard())
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

        config_data = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop_sequences": [self.stop] if isinstance(self.stop, str) else self.stop,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

        if self.response_format and self.response_format.get("type"):
            config_data["response_format"] = self.response_format["type"]

        if self.tools:
            config_data["tools"] = [tool.to_standard() for tool in self.tools]

        if self.tool_choice:
            if isinstance(self.tool_choice, str):
                config_data["tool_choice"] = ToolChoice(mode=self.tool_choice)
            elif isinstance(self.tool_choice, OpenAIToolChoice):
                config_data["tool_choice"] = self.tool_choice.to_standard()

        # Filter out None values to allow GenerateConfig defaults to apply
        config = GenerateConfig(**{k: v for k, v in config_data.items() if v is not None})

        return {
            "messages": standard_messages,
            "config": config,
            "system": system_message,
        }


class OpenAIResponseChoice(BaseModel):
    finish_reason: str
    index: int
    message: OpenAIAssistantMessage


class OpenAIResponseUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class OpenAIResponseBody(BaseModel):
    id: str
    choices: Annotated[List[OpenAIResponseChoice], Field(min_length=1)]
    created: int
    model: str
    system_fingerprint: Optional[str] = Field(default=None)
    object: Literal["chat.completion"]
    usage: Optional[OpenAIResponseUsage] = Field(default=None)

    def to_standard(self, model: str = None):
        tool_calls = None
        if self.choices[0].message.tool_calls:
            tool_calls = []
            for tool_call in self.choices[0].message.tool_calls:
                if tool_call:
                    tool_calls.append(tool_call.to_standard())

        return GenerationResult(
            model=model or self.model,
            stop_reason=self.choices[0].finish_reason,
            content=self.choices[0].message.content,
            reasoning_content=self.choices[0].message.reasoning_content,
            tool_calls=tool_calls,
            input_tokens=getattr(self.usage, "prompt_tokens", None),
            output_tokens=getattr(self.usage, "completion_tokens", None),
            total_tokens=getattr(self.usage, "total_tokens", None),
        )

    @classmethod
    def from_standard(cls, result: GenerationResult) -> "OpenAIResponseBody":
        message = OpenAIAssistantMessage.from_standard(result.to_message())
        choice = OpenAIResponseChoice(finish_reason=result.stop_reason, index=0, message=message)
        usage = None
        if (
            result.input_tokens is not None
            and result.output_tokens is not None
            and result.total_tokens is not None
        ):
            usage = OpenAIResponseUsage(
                prompt_tokens=result.input_tokens,
                completion_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
            )

        return cls(
            id=f"chatcmpl-{uuid4().hex}",
            choices=[choice],
            created=int(time.time()),
            model=result.model,
            object="chat.completion",
            usage=usage,
        )


@RemoteLanguageModel.register("openai-compatible")
class OpenAICompatibleModel(HttpServiceModel):
    META = RemoteLanguageModelMetaInfo(
        required_config_fields=["api_url", "is_visual_model", "is_tool_model"]
    )
    REQUEST_BODY_CLS = OpenAIRequestBody
    RESPONSE_BODY_CLS = OpenAIResponseBody

    def _make_api_headers(self):
        if self.config.api_key:
            return {"Authorization": f"Bearer {self.config.api_key.get_secret_value()}"}

        return None

    @classmethod
    @validate_call
    def _convert_message(cls, message: ChatMessage) -> OpenAIChatMessage:
        if isinstance(message, AssistantMessage):
            return OpenAIAssistantMessage.from_standard(message)
        elif isinstance(message, UserMessage):
            return OpenAIUserMessage.from_standard(message)
        elif isinstance(message, ToolMessage):
            return OpenAIToolMessage.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: Annotated[List[ChatMessage], Field(min_length=1)],
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = [self._convert_message(message) for message in messages]
        if system:
            messages = [OpenAISystemMessage(content=system)] + messages

        return {"messages": messages}

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [OpenAITool.from_standard(tool) for tool in tools]

        openai_tool_choice = None
        if tools and tool_choice is not None:
            openai_tool_choice = tool_choice.mode
            if openai_tool_choice == "any":
                if tool_choice.functions and len(tool_choice.functions) == 1:
                    openai_tool_choice = OpenAIToolChoice(
                        type="function", function={"name": tool_choice.functions[0]}
                    )
                else:
                    raise ValueError("OpenAI does not supported multi functions in `tool_choice`")

        return {"tools": tools, "tool_choice": openai_tool_choice}

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "frequency_penalty": config.frequency_penalty,
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "presence_penalty": config.presence_penalty,
            "response_format": {"type": config.response_format} if config.response_format else None,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
        }


@RemoteLanguageModel.register("openai")
class OpenAIModel(OpenAICompatibleModel):
    # reference: https://platform.openai.com/docs/models
    # TODO: gpt-4o-audio-preview/gpt-4o-audio-preview-2024-10-01
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.openai.com/v1/chat/completions",
        language_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-instruct",
            "gpt-3.5-turbo-instruct-0914",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-32k-0314",
            "o1-mini",
            "o1-mini-2024-09-12",
            "o1-preview",
            "o1-preview-2024-09-12",
            "o3-mini",
            "o3-mini-2025-01-31",
        ],
        visual_language_models=[
            "gpt-4-vision-preview",
            "gpt-4-0215-preview",
            "gpt-4-1106-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "chatgpt-4o-latest",
            "o1",
            "o1-2024-12-17",
        ],
        # https://platform.openai.com/docs/guides/function-calling
        tool_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-vision-preview",
            "gpt-4-0215-preview",
            "gpt-4-1106-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "o1",
            "o1-2024-12-17",
            "o1-preview",
            "o1-preview-2024-09-12",
            "o1-mini",
            "o1-mini-2024-09-12",
            "o3-mini",
            "o3-mini-2025-01-31",
        ],
        online_models=[
            # TODO
            # "gpt-4o-realtime-preview",
            # "gpt-4o-realtime-preview-2024-10-01",
        ],
        required_config_fields=["api_key"],
    )


class AzureOpenAIRequestBody(OpenAIRequestBody):
    model: Optional[str] = Field(default=None, exclude=True)


@RemoteLanguageModel.register("azure-openai")
class AzureOpenAIModel(OpenAICompatibleModel):
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions
    # noqa: https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2024-04-01-preview/inference.json
    _API_TEMPLATE = "{endpoint}/openai/deployments/{deployment_name}/chat/completions"

    # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/model-retirements#deprecated-models
    # "gpt-35-turbo-0613"/"gpt-35-turbo-0301" 已于 2025-02-13 过期,

    # NOTE: azure openai 的 deployment_name 可以和 model name 不一样，
    #       见 https://github.com/PrefectHQ/marvin/issues/842
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/model-retirements#current-models
    META = RemoteLanguageModelMetaInfo(
        language_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-instruct",
            "gpt-3.5-turbo-instruct-0914",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-32k-0314",
            "o1-mini",
            "o1-preview",
            "o3-mini",
        ],
        visual_language_models=[
            "gpt-4-vision-preview",
            "gpt-4-0215-preview",
            "gpt-4-1106-preview",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "o1",
        ],
        # https://platform.openai.com/docs/guides/function-calling
        tool_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-vision-preview",
            "gpt-4-0215-preview",
            "gpt-4-1106-preview",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "o1",
            "o1-preview",
            "o1-mini",
            "o3-mini",
        ],
        required_config_fields=[
            "api_key",
            "azure_endpoint",
            "azure_deployment_name",
        ],
    )
    REQUEST_BODY_CLS = AzureOpenAIRequestBody

    def _make_api_headers(self):
        return {"api-key": self.config.api_key.get_secret_value()}

    def _make_api_params(self):
        return {"api-version": self.config.azure_api_version}

    def _get_api_url(self):
        return self._API_TEMPLATE.format(
            endpoint=self.config.azure_endpoint,
            deployment_name=self.config.azure_deployment_name,
        )
