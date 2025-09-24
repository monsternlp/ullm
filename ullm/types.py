import mimetypes
from copy import deepcopy
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

import magic
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    Json,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    computed_field,
    model_validator,
)

# === Base ullm types ===


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    url: Optional[HttpUrl] = None
    path: Optional[str] = None
    mime_type: Optional[str] = None
    data: Optional[bytes] = None

    @model_validator(mode="before")
    @classmethod
    def check_image_data(cls, data):
        data = deepcopy(data)
        assert data.get("url") or data.get("path") or data.get("data")
        if data.get("path"):
            with open(data["path"], "rb") as image_file:
                data["data"] = image_file.read()

        if data.get("data"):
            data["mime_type"] = magic.from_buffer(data["data"], mime=True)
        else:
            data["mime_type"], _ = mimetypes.guess_type(str(data["url"]))

        return data


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: Annotated[List[Union[TextPart, ImagePart]], Field(min_length=1)]

    @model_validator(mode="before")
    @classmethod
    def validate_content(cls, data):
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            data = deepcopy(data)
            data["content"] = [TextPart(text=data["content"])]

        return data


class FunctionCall(BaseModel):
    name: str
    arguments: Optional[Union[Json, Dict[str, Any]]] = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: uuid4().hex)
    type: str
    function: Optional[FunctionCall] = None

    @model_validator(mode="after")
    def check_tool(self):
        if self.type == "function" and not self.function:
            raise ValueError("`function` should not be empty!")

        return self


class CitationSource(BaseModel):
    id: Optional[str] = None
    type: Literal["tool", "document"]
    tool_output: Optional[dict] = None
    document: Optional[dict] = None

    @model_validator(mode="before")
    @classmethod
    def check_type(cls, values):
        if values["type"] == "tool":
            assert values.get("tool_output")

        if values["type"] == "document":
            assert values.get("document")

        return values


class Citation(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None
    text: Optional[str] = None
    sources: Optional[List[CitationSource]] = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: List[TextPart | ImagePart] = Field(default_factory=list)
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    citations: Optional[List[Citation]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_content(cls, data):
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            data = deepcopy(data)
            data["content"] = [TextPart(text=data["content"])]
        return data

    @model_validator(mode="after")
    def check_content_or_tool_calls(self):
        assert self.content or self.tool_calls
        return self


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: Optional[str] = Field(default_factory=lambda: uuid4().hex)
    tool_name: str
    tool_result: str


ChatMessage = Union[UserMessage, AssistantMessage, ToolMessage]


class JsonSchemaObject(BaseModel):
    # https://spec.openapis.org/oas/v3.0.3#schema
    # https://ai.google.dev/api/rest/v1beta/Tool#Schema
    type: Literal["object"]
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    required: Optional[List[str]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def validate_json_schema(cls, data):
        try:
            import jsonschema

            jsonschema.Draft7Validator.check_schema(data)
            return data
        except (ValueError, Exception):
            raise ValueError("Not a valid json schema.")


class ParameterDefinition(BaseModel):
    type: str
    name: str
    description: str
    required: Optional[bool] = True


class FunctionObject(BaseModel):
    name: str
    description: str
    arguments: Optional[List[ParameterDefinition]] = Field(default_factory=list)
    returns: Optional[List[ParameterDefinition]] = Field(default_factory=list)


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionObject


class ToolChoice(BaseModel):
    # none: 不使用工具
    # auto: 模型自己判断
    # any: 从给定的工具中任选一个，如果不给定则从所有工具里选
    mode: Literal["none", "auto", "any"]
    functions: Optional[List[str]] = Field(default_factory=list)


class Thinking(BaseModel):
    type: Annotated[
        Literal["disabled", "enabled", "auto"],
        Field("disabled/enabled thinking, auto: let model decide"),
    ] = "auto"
    effort: Annotated[
        Literal["high", "medium", "low"] | None, Field("thinking effort, OpenAI Only")
    ] = None
    max_tokens: Annotated[int | None, Field("max thinking tokens, Google & Anthropic...")] = None
    exclude: Annotated[bool | None, Field("if True, exclude thinking content in response")] = False


class GenerateConfig(BaseModel):
    temperature: Optional[NonNegativeFloat] = None
    max_tokens: Optional[PositiveInt] = None
    max_input_tokens: Optional[PositiveInt] = None
    max_output_tokens: Optional[PositiveInt] = None
    top_p: Optional[NonNegativeFloat] = None
    top_k: Optional[NonNegativeInt] = None
    stop_sequences: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    modalities: Annotated[
        List[Literal["text", "audio", "image"]] | None,
        Field("Output types the model expected to generate."),
    ] = None
    response_format: Optional[Literal["text", "json_object"]] = "text"
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    extra: Optional[dict] = None
    thinking: Optional[Thinking] = None


class GenerationResult(BaseModel):
    model: str
    stop_reason: str
    message: AssistantMessage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    original_result: Json[Any] = None

    @model_validator(mode="before")
    @classmethod
    def build_message_from_legacy_fields(cls, data):
        """
        兼容旧设计：允许初始化时传入
        content/tool_calls/reasoning_content/citations
        并转为 message
        """
        if not isinstance(data, dict):
            return data

        data = deepcopy(data)
        legacy_content = data.pop("content", None)
        legacy_tool_calls = data.pop("tool_calls", None)
        legacy_reasoning_content = data.pop("reasoning_content", None)
        legacy_citations = data.pop("citations", None)

        if data.get("message") is not None:
            return data

        message_payload = {}
        content_parts = []
        if isinstance(legacy_content, str):
            if legacy_content != "":
                content_parts.append({"type": "text", "text": legacy_content})
            message_payload["content"] = content_parts

        if legacy_tool_calls:
            message_payload["tool_calls"] = legacy_tool_calls
        if legacy_reasoning_content:
            message_payload["reasoning_content"] = legacy_reasoning_content
        if legacy_citations:
            message_payload["citations"] = legacy_citations

        data["message"] = message_payload

        return data

    def _filter_parts(self, part_type) -> List:
        if not self.message or not self.message.content:
            return []
        return [part for part in self.message.content if isinstance(part, part_type)]

    @computed_field(return_type=List[ImagePart])
    def images(self) -> List[ImagePart]:
        """
        从 message 过滤得到的所有 ImagePart
        """
        return self._filter_parts(ImagePart)

    @computed_field(return_type=List[TextPart])
    def texts(self) -> List[TextPart]:
        """
        从 message 过滤得到的所有 TextPart
        """
        return self._filter_parts(TextPart)

    @computed_field(return_type=str)
    def content(self) -> str:
        """
        全部文本信息
        """
        return "\n".join([text.text for text in self._filter_parts(TextPart)])

    @computed_field(return_type=List[ToolCall] | None)
    def tool_calls(self) -> List[ToolCall] | None:
        return self.message.tool_calls

    @computed_field(return_type=str | None)
    def reasoning_content(self) -> str | None:
        return self.message.reasoning_content

    @computed_field(return_type=List[Citation] | None)
    def citations(self) -> List[Citation] | None:
        return self.message.citations

    def to_message(self) -> AssistantMessage:
        if self.message:
            return self.message
        return AssistantMessage(
            tool_calls=self.tool_calls,
            citations=self.citations,
        )
