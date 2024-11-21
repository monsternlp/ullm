import base64
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, confloat, conlist, model_validator, validate_call

from .base import (
    AssistantMessage,
    ChatMessage,
    FunctionCall,
    GenerateConfig,
    GenerationResult,
    HttpServiceModel,
    ImagePart,
    JsonSchemaObject,
    PositiveInt,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    TextPart,
    Tool,
    ToolCall,
    ToolChoice,
    ToolMessage,
    UserMessage,
)


class AnthropicTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicImageSource(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class AnthropicImagePart(BaseModel):
    type: Literal["image"] = "image"
    source: AnthropicImageSource

    @classmethod
    def from_standard(cls, part: ImagePart):
        if part.data:
            return cls(
                source=AnthropicImageSource(
                    media_type=part.mime_type, data=base64.b64encode(part.data).decode("utf-8")
                )
            )

        return None


class AnthropicToolUsePart(BaseModel):
    # https://docs.anthropic.com/claude/docs/tool-use#tool-use-and-tool-result-content-blocks
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class AnthropicToolResultPart(BaseModel):
    # https://docs.anthropic.com/claude/docs/tool-use#tool-use-and-tool-result-content-blocks
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[str] = None
    is_error: Optional[bool] = None


class AnthropicChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        conlist(
            Union[
                AnthropicTextPart, AnthropicImagePart, AnthropicToolUsePart, AnthropicToolResultPart
            ],
            min_length=1,
        ),
    ]

    @classmethod
    def from_standard(cls, message: ChatMessage):
        if isinstance(message, UserMessage):
            if isinstance(message.content, str):
                return cls(role="user", content=message.content)

            parts = []
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append(AnthropicTextPart(text=part.text))
                elif isinstance(part, ImagePart):
                    image_part = AnthropicImagePart.from_standard(part)
                    if image_part:
                        parts.append(image_part)
                    else:
                        pass

            return cls(role="user", content=parts)
        if isinstance(message, AssistantMessage):
            content = message.content
            if message.tool_calls:
                content = [{"type": "text", "text": content}] if message.content else []
                for tool_call in message.tool_calls:
                    content.append(
                        AnthropicToolUsePart(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            input=tool_call.function.arguments,
                        )
                    )

            return cls(role="assistant", content=content)
        if isinstance(message, ToolMessage):
            return cls(
                role="user",
                content=[
                    AnthropicToolResultPart(
                        tool_use_id=message.tool_call_id, content=message.tool_result
                    )
                ],
            )


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[JsonSchemaObject] = None

    @classmethod
    def from_standard(cls, tool: Tool):
        properties, required = {}, []
        for argument in tool.function.arguments or []:
            if argument.required:
                required.append(argument.name)

            properties[argument.name] = {
                "type": argument.type,
                "description": argument.description,
            }

        return cls(
            name=tool.function.name,
            description=tool.function.description,
            input_schema={"type": "object", "properties": properties, "required": required},
        )


class AnthropicToolChoice(BaseModel):
    type: Literal["auto", "any", "tool"]
    disable_parallel_tool_use: Optional[bool] = None
    name: Optional[str] = None


class AnthropicRequestMeta(BaseModel):
    user_id: str


class AnthropicRequestBody(BaseModel):
    model: str
    messages: conlist(AnthropicChatMessage, min_length=1)
    max_tokens: Optional[int] = None
    metadata: Optional[AnthropicRequestMeta] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = None
    system: Optional[str] = None
    temperature: Optional[confloat(ge=0.0, le=1.0)] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    tools: Optional[List[AnthropicTool]] = None
    top_k: Optional[PositiveInt] = None
    top_p: Optional[confloat(ge=0.0, le=1.0)] = None

    @model_validator(mode="after")
    def check_messages(self):
        assert self.messages[0].role == "user"
        return self


class AnthropicResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicResponseBody(BaseModel):
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    content: List[Union[AnthropicTextPart, AnthropicToolUsePart]]
    model: str
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    stop_sequence: Optional[str] = None
    usage: AnthropicResponseUsage

    def to_standard(self, model: str = None):
        content, tool_calls = "", []
        for part in self.content:
            if isinstance(part, AnthropicTextPart):
                content = content + "\n" + part.text
            else:
                tool_calls.append(
                    ToolCall(
                        id=part.id,
                        type="function",
                        function=FunctionCall(name=part.name, arguments=part.input),
                    )
                )

        return GenerationResult(
            model=model or self.model,
            stop_reason=self.stop_reason,
            content=content,
            tool_calls=tool_calls,
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            total_tokens=self.usage.input_tokens + self.usage.output_tokens,
        )


@RemoteLanguageModel.register("anthropic")
class AnthropicModel(HttpServiceModel):
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.anthropic.com/v1/messages",
        visual_language_models=[
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-20241022",
            "claude-3-5-haiku-latest",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-opus-latest",
            "claude-3-sonnet-20240229",
        ],
        tool_models=[
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-20241022",
            "claude-3-5-haiku-latest",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-opus-latest",
            "claude-3-sonnet-20240229",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = AnthropicRequestBody
    RESPONSE_BODY_CLS = AnthropicResponseBody
    # reference: https://docs.anthropic.com/claude/reference/versions
    _ANTHROPIC_VERSION = "2023-06-01"

    def _make_api_headers(self):
        return {
            "x-api-key": self.config.api_key.get_secret_value(),
            "anthropic-version": self._ANTHROPIC_VERSION,
        }

    @classmethod
    @validate_call
    def _convert_message(cls, message: ChatMessage):
        return AnthropicChatMessage.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = [self._convert_message(message) for message in messages]
        if system:
            return {"messages": messages, "system": system}

        return {"messages": messages}

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [AnthropicTool.from_standard(tool) for tool in tools]

        anthropic_tool_choice = None
        if tools and tool_choice is not None and tool_choice != "none":
            anthropic_tool_choice = {"type": tool_choice}
            if anthropic_tool_choice == "any":
                if len(tool_choice.functions) == 1:
                    anthropic_tool_choice = {"type": "tool", "name": tool_choice.functions[0]}
                else:
                    raise ValueError(
                        "Anthropic does not supported multi functions in `tool_choice`"
                    )

        return {"tools": tools, "tool_choice": anthropic_tool_choice}

    @validate_call
    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "stop_sequences": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
            "top_k": config.top_k or self.config.top_k,
        }
