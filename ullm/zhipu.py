import base64
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    confloat,
    conlist,
    model_validator,
    validate_call,
)

from .base import (
    ChatMessage,
    GenerateConfig,
    ImagePart,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    TextPart,
    Tool,
    ToolChoice,
    UserMessage,
)
from .openai import (
    OpenAIAssistantMessage,
    OpenAIChatMessage,
    OpenAICompatibleModel,
    OpenAIFunctionObject,
    OpenAIImagePart,
    OpenAIImageURL,
    OpenAIRequestBody,
    OpenAIResponseBody,
    OpenAISystemMessage,
    OpenAIToolMessage,
    OpenAIUserMessage,
)


class ZhipuUserMessage(OpenAIUserMessage):
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
                url = None
                if part.data:
                    url = base64.b64encode(part.data).decode("utf-8")
                elif part.url:
                    url = part.url

                if url:
                    parts.append(OpenAIImagePart(image_url=OpenAIImageURL(url=str(url))))

        return cls(content=parts)


ZhipuChatMessage = Union[
    OpenAISystemMessage, ZhipuUserMessage, OpenAIToolMessage, OpenAIAssistantMessage
]


class ZhipuRetrievalTool(BaseModel):
    knowledge_id: str
    prompt_template: Optional[str] = None


class ZhipuWebSearchTool(BaseModel):
    enable: Optional[bool] = False
    search_query: Optional[str] = None
    search_result: Optional[bool] = False
    search_prompt: Optional[str] = None


class ZhipuAITool(BaseModel):
    type: Literal["function", "web_search", "retrieval"]
    function: Optional[OpenAIFunctionObject] = None
    web_search: Optional[ZhipuWebSearchTool] = None
    retrieval: Optional[ZhipuRetrievalTool] = None

    @model_validator(mode="before")
    @classmethod
    def check_type(cls, data):
        assert data.get(data["type"]) is not None
        return data

    @classmethod
    def from_standard(cls, tool: Tool):
        return cls(type=tool.type, function=OpenAIFunctionObject.from_standard(tool.function))


class ZhipuAIRequestBody(OpenAIRequestBody):
    # reference: https://open.bigmodel.cn/dev/api
    ## exclude fields
    frequency_penalty: Optional[Any] = Field(None, exclude=True)
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(1, exclude=True)
    presence_penalty: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    seed: Optional[int] = Field(None, exclude=True)
    user: Optional[str] = Field(None, exclude=True)

    ## different parameters
    messages: conlist(ZhipuChatMessage, min_length=1)
    temperature: Optional[confloat(gt=0.0, lt=1.0)] = None
    top_p: Optional[confloat(gt=0.0, lt=1.0)] = None
    stop: Optional[List[str]]
    tools: Optional[List[ZhipuAITool]] = None
    tool_choice: Optional[Literal["auto"]] = "auto"

    ## ZhipuAI-specific parameters
    request_id: Optional[str] = None
    do_sample: Optional[bool] = True
    user_id: Optional[str] = None


class ZhipuAIResponseBody(OpenAIResponseBody):
    system_fingerprint: Optional[str] = Field(None, exclude=True)
    object: Optional[Literal["chat.completion"]] = Field(None, exclude=True)


@RemoteLanguageModel.register("zhipu")
class ZhipuAIModel(OpenAICompatibleModel):
    # reference: https://open.bigmodel.cn/dev/api
    META = RemoteLanguageModelMetaInfo(
        api_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        language_models=[
            "glm-4-plus",
            "glm-4-plus-online",
            "glm-4-0520",
            "glm-4-0520-online",
            "glm-4",
            "glm-4-online",
            "glm-4-air",
            "glm-4-air-online",
            "glm-4-long",
            "glm-4-long-online",
            "glm-4-flash",
            "glm-4-flash-online",
            "glm-zero-preview",
        ],
        visual_language_models=["glm-4v", "glm-4v-plus"],
        tool_models=[
            "glm-4-plus",
            "glm-4-plus-online",
            "glm-4-0520",
            "glm-4-0520-online",
            "glm-4",
            "glm-4-online",
            "glm-4-air",
            "glm-4-air-online",
            "glm-4-long",
            "glm-4-long-online",
            "glm-4-flash",
            "glm-4-flash-online",
        ],
        online_models=[
            "glm-4-plus-online",
            "glm-4-0520-online",
            "glm-4-online",
            "glm-4-air-online",
            "glm-4-long-online",
            "glm-4-flash-online",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = ZhipuAIRequestBody
    RESPONSE_BODY_CLS = ZhipuAIResponseBody

    @classmethod
    @validate_call
    def _convert_message(cls, message: ChatMessage) -> OpenAIChatMessage:
        if isinstance(message, UserMessage):
            return ZhipuUserMessage.from_standard(message)
        else:
            return super()._convert_message(message)

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [ZhipuAITool.from_standard(tool) for tool in tools]

        if self.is_online_model():
            tools = tools or []
            tools.append(
                ZhipuAITool(
                    type="web_search",
                    web_search=ZhipuWebSearchTool(enable=True, search_result=True),
                )
            )
        else:
            tools = tools or []
            tools.append(
                ZhipuAITool(type="web_search", web_search=ZhipuWebSearchTool(enable=False))
            )

        return {"tools": tools}

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "model": self.model.replace("-online", ""),
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
        }
