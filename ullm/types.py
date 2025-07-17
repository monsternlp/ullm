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
    def validate_content(cls, data):
        data = deepcopy(data)
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            data["content"] = [TextPart(text=data["content"])]

        return data


class FunctionCall(BaseModel):
    name: str
    arguments: Optional[Union[Json, Dict[str, Any]]] = {}


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
    content: Optional[str] = ""
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    citations: Optional[List[Citation]] = None

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
    properties: Optional[Dict[str, Any]] = {}
    required: Optional[List[str]] = []

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
    arguments: Optional[List[ParameterDefinition]] = []
    returns: Optional[List[ParameterDefinition]] = []


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionObject


class ToolChoice(BaseModel):
    # none: 不使用工具
    # auto: 模型自己判断
    # any: 从给定的工具中任选一个，如果不给定则从所有工具里选
    mode: Literal["none", "auto", "any"]
    functions: Optional[List[str]] = []


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
    response_format: Optional[Literal["text", "json_object"]] = "text"
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    extra: Optional[dict] = None
    thinking_type: Optional[Literal["disabled", "enabled", "auto"]] = None


class GenerationResult(BaseModel):
    model: str
    stop_reason: str
    content: Optional[str] = ""
    reasoning_content: Optional[str] = ""
    tool_calls: Optional[List[ToolCall]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    original_result: Json[Any] = None
    citations: Optional[List[Citation]] = None

    @model_validator(mode="after")
    def check_content_or_tool_calls(self):
        assert self.content or self.tool_calls
        return self

    def to_message(self) -> AssistantMessage:
        return AssistantMessage(
            content=self.content, tool_calls=self.tool_calls, citations=self.citations
        )
