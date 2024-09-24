from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conint, conlist, model_validator, validate_call

from .base import (
    AssistantMessage,
    ChatMessage,
    GenerateConfig,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    TextPart,
    Tool,
    ToolChoice,
    ToolMessage,
    UserMessage,
)
from .openai import (
    OpenAICompatibleModel,
    OpenAIFunctionObject,
    OpenAIRequestBody,
    OpenAIResponseBody,
    OpenAIToolCall,
)


class BaichuanChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    # NOTE: https://platform.baichuan-ai.com/docs/api#12 文档上说 tool_calls 是 string 类型
    #       但将 tool_calls json.dumps 后调用 API 会报错，按 OpenAI 的形式传入则正常
    tool_calls: Optional[List[OpenAIToolCall]] = None
    tool_call_id: Optional[str] = None

    @classmethod
    def from_standard(cls, message: ChatMessage):
        if isinstance(message, AssistantMessage):
            content = message.content
            tool_calls = None
            if message.tool_calls:
                content, tool_calls = "", []
                for tool_call in message.tool_calls:
                    tool_calls.append(OpenAIToolCall.from_standard(tool_call))

            return cls(role="assistant", content=content.strip(), tool_calls=tool_calls)

        if isinstance(message, UserMessage):
            content = ""
            for part in message.content:
                if isinstance(part, TextPart):
                    content += part.text + "\n"
                else:
                    pass

            return cls(role="user", content=content.strip())

        if isinstance(message, ToolMessage):
            return cls(role="tool", content=message.tool_result, tool_call_id=message.tool_call_id)


class BaichuanRetrievalObject(BaseModel):
    kb_ids: List[str]
    answer_mode: Optional[Literal["knowledge-base-only"]] = None


class BaichuanWebSearchObject(BaseModel):
    enable: Optional[bool] = False
    search_mode: Optional[Literal["performance_first", "quality_first"]] = "performance_first"


BaichuanFunctionObject = OpenAIFunctionObject


class BaichuanTool(BaseModel):
    type: Optional[Literal["web_search", "retrieval", "function"]] = None
    retrieval: Optional[BaichuanRetrievalObject] = None
    web_search: Optional[BaichuanWebSearchObject] = None
    function: Optional[BaichuanFunctionObject] = None

    @classmethod
    def from_standard(cls, tool: Tool):
        return cls(type=tool.type, function=BaichuanFunctionObject.from_standard(tool.function))


class BaichuanRequestBody(OpenAIRequestBody):
    # reference: https://platform.baichuan-ai.com/docs/api#12
    ## excluded parameters
    frequency_penalty: Optional[Any] = Field(None, exclude=True)
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(None, exclude=True)
    presence_penalty: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    stop: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)

    ## different parameters
    messages: conlist(BaichuanChatMessage, min_length=1)
    tools: Optional[List[BaichuanTool]] = None
    tool_choice: Optional[Literal["auto", "none"]] = None

    ## Baichuan-specific parameters
    top_k: Optional[conint(ge=0, le=20)] = None


class BaichuanResponseBody(OpenAIResponseBody):
    @model_validator(mode="before")
    @classmethod
    def check_usage(cls, values):
        if "usage" not in values:
            values["usage"] = {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
            }

        return values


@RemoteLanguageModel.register("baichuan")
class BaichuanModel(OpenAICompatibleModel):
    # reference: https://platform.baichuan-ai.com/docs/api
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.baichuan-ai.com/v1/chat/completions",
        language_models=[
            "Baichuan2-Turbo",
            "Baichuan2-Turbo-online",
            "Baichuan3-Turbo",
            "Baichuan3-Turbo-128k",
            "Baichuan3-Turbo-online",
            "Baichuan3-Turbo-128k-online",
            "Baichuan4",
            "Baichuan4-online",
            "Baichuan4-Turbo",
            "Baichuan4-Turbo-online",
            "Baichuan4-Air",
            "Baichuan4-Air-online",
        ],
        visual_language_models=[],
        tool_models=[
            "Baichuan3-Turbo",
            "Baichuan3-Turbo-128k",
            "Baichuan4",
            "Baichuan4-Turbo",
            "Baichuan4-Air",
        ],
        online_models=[
            "Baichuan2-Turbo-online",
            "Baichuan3-Turbo-online",
            "Baichuan3-Turbo-128k-online",
            "Baichuan4-online",
            "Baichuan4-Turbo-online",
            "Baichuan4-Air-online",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = BaichuanRequestBody
    RESPONSE_BODY_CLS = BaichuanResponseBody

    @classmethod
    @validate_call
    def _convert_message(cls, message: ChatMessage) -> BaichuanChatMessage:
        return BaichuanChatMessage.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = [self._convert_message(message) for message in messages]
        if system:
            messages = [BaichuanChatMessage(role="system", content=system)] + messages

        return {"messages": messages}

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [BaichuanTool.from_standard(tool) for tool in tools]

        baichuan_tool_choice = None
        if tools and tool_choice is not None:
            baichuan_tool_choice = tool_choice.mode
            if baichuan_tool_choice == "any":
                baichuan_tool_choice = "auto"

        return {"tools": tools, "tool_choice": baichuan_tool_choice}

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        generation_config = {
            "model": self.model.replace("-online", ""),
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
            "top_k": config.top_k or self.config.top_k,
        }
        if self.is_online_model():
            generation_config["tools"] = [{"type": "web_search", "web_search": {"enable": True}}]

        return generation_config
