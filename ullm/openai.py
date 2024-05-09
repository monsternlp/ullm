import base64
import json
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, confloat, conint, conlist, validate_call

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

OpenAITextPart = TextPart


class OpenAISystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class OpenAIImageURL(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class OpenAIImagePart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: OpenAIImageURL


class OpenAIUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: Union[str, conlist(Union[OpenAITextPart, OpenAIImagePart], min_length=1)]

    @classmethod
    def from_standard(cls, user_message: UserMessage):
        if all(isinstance(part, TextPart) for part in user_message.content):
            content = "\n".join([part.text for part in user_message.content])
            return cls(content=content)

        parts = []
        for part in user_message.content:
            if isinstance(part, TextPart):
                parts.append(part)
            elif isinstance(part, ImagePart):
                if part.url:
                    parts.append(OpenAIImagePart(image_url=OpenAIImageURL(url=str(part.url))))

                else:
                    base64_data = base64.b64encode(part.data).decode("utf-8")
                    parts.append(
                        OpenAIImagePart(
                            image_url=OpenAIImageURL(
                                url=f"data:{part.mime_type};base64,{base64_data}"
                            )
                        )
                    )

        return cls(content=parts)


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


class OpenAIAssistantMessage(AssistantMessage):
    tool_calls: Optional[List[OpenAIToolCall]] = None

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


class OpenAIToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: Optional[str] = "success"
    tool_call_id: Optional[str] = Field(default_factory=lambda: uuid4().hex)

    @classmethod
    def from_standard(cls, tool_message: ToolMessage):
        return cls(
            role=tool_message.role,
            content=tool_message.tool_result,
            tool_call_id=tool_message.tool_call_id,
        )


OpenAIChatMessage = Union[
    OpenAISystemMessage, OpenAIUserMessage, OpenAIToolMessage, OpenAIAssistantMessage
]


class OpenAIFunctionObject(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[JsonSchemaObject] = None

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


class OpenAITool(BaseModel):
    type: Literal["function"]
    function: OpenAIFunctionObject

    @classmethod
    def from_standard(cls, tool: Tool):
        return cls(type=tool.type, function=OpenAIFunctionObject.from_standard(tool.function))


class OpenAIToolChoice(BaseModel):
    type: Literal["function"]
    function: Dict[Literal["name"], str]


class OpenAIRequestBody(BaseModel):
    # https://platform.openai.com/docs/api-reference/chat/create
    # NOTE:
    # 1. gpt-4-vision 不能设置 logprobs/logit_bias/tools/tool_choice/response_format 几个参数
    # 2. 只有 gpt-4-turbo 系列模型和比 gpt-3.5-turbo-1106 更新的模型可以使用 response_format 参数
    messages: conlist(OpenAIChatMessage, min_length=1)
    model: str
    frequency_penalty: Optional[confloat(ge=-2.0, le=2.0)] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[conint(ge=0, le=20)] = None
    max_tokens: Optional[int] = None
    n: Optional[conint(ge=1, le=128)] = 1
    presence_penalty: Optional[confloat(ge=-2.0, le=2.0)] = None
    response_format: Optional[Dict[Literal["type"], Literal["text", "json_object"]]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[confloat(ge=0.0, le=2.0)] = None
    top_p: Optional[confloat(ge=0.0, le=1.0)] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[Literal["auto", "none"], OpenAIToolChoice]] = None
    user: Optional[str] = None


class OpenAIResponseChoice(BaseModel):
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"]
    index: int
    message: OpenAIAssistantMessage


class OpenAIResponseUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class OpenAIResponseBody(BaseModel):
    id: str
    choices: conlist(OpenAIResponseChoice, min_length=1)
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    object: Literal["chat.completion"]
    usage: Optional[OpenAIResponseUsage] = None

    def to_standard(self, model: str = None):
        tool_calls = None
        if self.choices[0].message.tool_calls:
            tool_calls = []
            for tool_call in self.choices[0].message.tool_calls:
                tool_calls.append(ToolCall.model_validate(tool_call.model_dump()))

        return GenerationResult(
            model=model or self.model,
            stop_reason=self.choices[0].finish_reason,
            content=self.choices[0].message.content,
            tool_calls=tool_calls,
            input_tokens=self.usage.prompt_tokens,
            output_tokens=self.usage.completion_tokens,
            total_tokens=self.usage.total_tokens,
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
        messages: conlist(ChatMessage, min_length=1),
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
                if len(tool_choice.functions) == 1:
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
    # reference: https://platform.openai.com/docs/models/overview
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.openai.com/v1/chat/completions",
        language_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-0613",  # Will be deprecated on June 13, 2024.
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",  # Currently points to gpt-3.5-turbo-16k-0613.
            "gpt-3.5-turbo-16k-0613",  # Will be deprecated on June 13, 2024.
            "gpt-3.5-turbo-instruct",
            "gpt-4",
            "gpt-4-0125-preview",
            "gpt-4-0613",
            "gpt-4-1106-preview",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
        ],
        visual_language_models=[
            "gpt-4-vision-preview",
            "gpt-4-1106-vision-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
        ],
        # https://platform.openai.com/docs/guides/function-calling
        tool_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-4",
            "gpt-4-0125-preview",
            "gpt-4-0613",
            "gpt-4-1106-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
        ],
        required_config_fields=["api_key"],
    )


class AzureOpenAIRequestBody(OpenAIRequestBody):
    model: Optional[str] = Field(None, exclude=True)


@RemoteLanguageModel.register("azure-openai")
class AzureOpenAIModel(OpenAICompatibleModel):
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions
    # noqa: https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2024-04-01-preview/inference.json
    _API_TEMPLATE = "{endpoint}/openai/deployments/{deployment_name}/chat/completions"

    # NOTE: azure openai 的 deployment_name 可以和 model name 不一样，
    #       见 https://github.com/PrefectHQ/marvin/issues/842
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models
    META = RemoteLanguageModelMetaInfo(
        language_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-0613",  # Will be deprecated on June 13, 2024.
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",  # Currently points to gpt-3.5-turbo-16k-0613.
            "gpt-3.5-turbo-16k-0613",  # Will be deprecated on June 13, 2024.
            "gpt-3.5-turbo-instruct",
            "gpt-4",
            "gpt-4-0125-preview",
            "gpt-4-0613",
            "gpt-4-1106-preview",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
        ],
        visual_language_models=[
            "gpt-4-vision-preview",
            "gpt-4-1106-vision-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
        ],
        # https://platform.openai.com/docs/guides/function-calling
        tool_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-4",
            "gpt-4-0125-preview",
            "gpt-4-0613",
            "gpt-4-1106-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
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
