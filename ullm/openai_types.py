import base64
import json
import time
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl

from .types import (
    AssistantMessage,
    ChatMessage,
    FunctionObject,
    GenerateConfig,
    GenerationResult,
    ImagePart,
    JsonSchemaObject,
    TextPart,
    Thinking,
    Tool,
    ToolCall,
    ToolChoice,
    ToolMessage,
    UserMessage,
)

# === OpenAI types ===


class OpenAISystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str

    def to_standard(self) -> str:
        return self.content


class OpenAIImageURL(BaseModel):
    url: HttpUrl | str = Field(description="Data URL(base64) or URL of the image")
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


class OpenAIAssistantMessage(BaseModel):
    # Strictly follow OpenAI Chat Completions message schema for assistant
    # https://platform.openai.com/docs/api-reference/chat/object
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

    # Extra fields from openrouter
    # https://openrouter.ai/docs/features/multimodal/image-generation
    images: Optional[List[OpenAIImagePart]] = None
    # NOTE:
    #   - openai 文档中未观察到 reasoning 相关返回值
    #   - openrouter 返回值中同时有 reasoning 和 reasoning_details
    #     - reasoning: 未在文档中观察到
    #     - reasoning_details: https://openrouter.ai/docs/use-cases/reasoning-tokens#reasoning_details-array-structure
    # TODO: reasoning_details -> reasoning
    reasoning: Optional[Any] = None

    @classmethod
    def from_standard(cls, message: AssistantMessage):
        tool_calls = None
        if message.tool_calls:
            tool_calls = [OpenAIToolCall.from_standard(tc) for tc in message.tool_calls]

        return cls(role="assistant", content=message.content, tool_calls=tool_calls)

    def to_standard(self) -> AssistantMessage:
        parts: List[TextPart | ImagePart] = []

        if isinstance(self.content, str) and self.content != "":
            parts.append(TextPart(text=self.content))

        # NOTE: Only for OpenRouter: https://openrouter.ai/docs/features/multimodal/image-generation
        if self.images:
            parts.extend(image.to_standard() for image in self.images)

        tool_calls: Optional[List[ToolCall]] = None
        if self.tool_calls:
            tool_calls = [tc.to_standard() for tc in self.tool_calls]

        return AssistantMessage(content=parts, tool_calls=tool_calls)


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


class OpenRouterReasoning(BaseModel):
    """
    https://openrouter.ai/docs/use-cases/reasoning-tokens#reasoning-effort-level
    NOTE: Only supported by openrouter.
    Configuration for model reasoning/thinking tokens
    """

    effort: Annotated[
        Literal["high", "medium", "low"] | None,
        Field(description="OpenAI-style reasoning effort setting"),
    ] = None
    max_tokens: Annotated[
        int | None,
        Field(
            description="Non-OpenAI-style reasoning effort setting. Cannot be used simultaneously with effort."  # noqa: E501
        ),
    ] = None
    exclude: Annotated[
        bool | None, Field(description="Whether to exclude reasoning from the response")
    ] = False

    @classmethod
    def from_standard(cls, thinking: Thinking):
        return cls(
            effort=thinking.effort,
            max_tokens=thinking.max_tokens,
            exclude=thinking.exclude,
        )


class OpenAIRequestBody(BaseModel):
    # https://platform.openai.com/docs/api-reference/chat/create
    # https://openrouter.ai/docs/api-reference/chat-completion
    # NOTE:
    # 0. 受实际使用场景影响，目前优先适配 openrouter api 而不是 openai
    # 1. gpt-4-vision 不能设置 logprobs/logit_bias/tools/tool_choice/response_format 几个参数
    # 2. 只有 gpt-4-turbo 系列模型和比 gpt-3.5-turbo-1106 更新的模型可以使用 response_format 参数
    # 3. `reasoning` 只被 openrouter 支持，openai api 中对应参数为 `reasoning_effort`
    # TODO: 严格区分 openai 和 openrouter 的差异参数
    messages: Annotated[List[OpenAIChatMessage], Field(min_length=1)]
    model: str
    frequency_penalty: Optional[Annotated[float, Field(ge=-2.0, le=2.0)]] = Field(default=None)
    logit_bias: Optional[Dict[str, int]] = Field(default=None)
    logprobs: Optional[bool] = Field(default=None)
    top_logprobs: Optional[Annotated[int, Field(ge=0, le=20)]] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    n: Optional[Annotated[int, Field(ge=1, le=128)]] = Field(default=1)
    modalities: Optional[List[Literal["text", "audio", "image"]]] = None
    presence_penalty: Optional[Annotated[float, Field(ge=-2.0, le=2.0)]] = Field(default=None)
    reasoning: Optional[OpenRouterReasoning] = Field(default=None)
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

    def to_standard(self, model: str = None) -> GenerationResult:
        assistant_message = self.choices[0].message.to_standard()
        return GenerationResult(
            model=model or self.model,
            stop_reason=self.choices[0].finish_reason,
            message=assistant_message,
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


class AzureOpenAIRequestBody(OpenAIRequestBody):
    model: Optional[str] = Field(default=None, exclude=True)
