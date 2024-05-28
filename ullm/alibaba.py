from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    PositiveInt,
    confloat,
    conlist,
    model_validator,
    validate_call,
)

from .base import (
    AssistantMessage,
    ChatMessage,
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
from .openai import OpenAITool, OpenAIToolCall


class AlibabaContentPart(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None

    @model_validator(mode="before")
    def check_values(cls, values):
        if values["role"] == "tool":
            assert values.get("name")

        return values


class AlibabaChatMessage(BaseModel):
    name: Optional[str] = None
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[AlibabaContentPart]]] = ""
    tool_calls: Optional[List[OpenAIToolCall]] = None

    @model_validator(mode="before")
    def check_values(cls, values):
        assert not values.get("tool_calls") or values["role"] == "assistant"
        if values["role"] != "assistant" and not values.get("tool_calls"):
            assert values.get("content") or values.get("contents")

        if values["role"] == "tool":
            assert values.get("name")

        return values

    @classmethod
    def from_standard(cls, message: ChatMessage):
        if isinstance(message, UserMessage):
            if all(isinstance(part, TextPart) for part in message.content):
                content = "\n".join([part.text for part in message.content])
                return cls(role=message.role, content=content)

            parts = []
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append(AlibabaContentPart(text=part.text))
                elif isinstance(part, ImagePart):
                    if part.url:
                        parts.append(AlibabaContentPart(image=str(part.url)))
                    else:
                        raise NotImplementedError

            return cls(role=message.role, content=parts)
        if isinstance(message, AssistantMessage):
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
        if isinstance(message, ToolMessage):
            return cls(role=message.role, name=message.tool_name, content=message.tool_result)


class AlibabaInputObject(BaseModel):
    messages: conlist(AlibabaChatMessage, min_length=1)


class AlibabaGenerateParameters(BaseModel):
    # reference:
    # https://help.aliyun.com/zh/dashscope/developer-reference/api-details?spm=a2c4g.11186623.0.0.4070e0f6LPgURw#b8ebf6b25eul6
    # 进行中的 OpenAI 接口兼容: https://help.aliyun.com/zh/dashscope/developer-reference/compatibility-of-openai-with-dashscope
    result_format: Optional[Literal["text", "message"]] = "message"
    seed: Optional[int] = None
    max_tokens: Optional[PositiveInt] = None
    top_p: Optional[confloat(gt=0.0, lt=1.0)] = None
    top_k: Optional[PositiveInt] = None
    repetition_penalty: Optional[float] = None
    temperature: Optional[confloat(ge=0, lt=2.0)] = None
    stop: Optional[Union[str, List[str]]] = None
    enable_search: Optional[bool] = None
    incremental_output: Optional[bool] = None
    tools: Optional[List[OpenAITool]] = None


class AlibabaRequestBody(BaseModel):
    model: str
    input: AlibabaInputObject
    parameters: AlibabaGenerateParameters


class AlibabaResponseChoice(BaseModel):
    finish_reason: Literal["null", "stop", "length", "tool_calls"]
    message: AlibabaChatMessage


class AlibabaResponseOutput(BaseModel):
    text: Optional[str] = None
    finish_reason: Optional[str] = None
    choices: Optional[List[AlibabaResponseChoice]] = None

    @model_validator(mode="before")
    def check_text_or_choices(cls, values):
        assert values.get("text") or values.get("choices")
        return values

    def to_choices(self):
        if self.choices:
            return self.choices

        return [
            AlibabaResponseChoice(
                finish_reason=self.finish_reason,
                message=AlibabaChatMessage(role="assistant", content=self.text),
            )
        ]


class AlibabaResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class AlibabaResponseBody(BaseModel):
    code: Optional[str] = ""
    message: Optional[str] = ""
    request_id: str
    output: AlibabaResponseOutput
    usage: AlibabaResponseUsage

    def to_standard(self, model: str = None):
        choices = self.output.to_choices()

        tool_calls = None
        if choices[0].message.tool_calls:
            tool_calls = []
            for tool_call in choices[0].message.tool_calls:
                tool_calls.append(ToolCall.model_validate(tool_call.model_dump()))

        return GenerationResult(
            model=model,
            stop_reason=choices[0].finish_reason,
            content=choices[0].message.content,
            tool_calls=tool_calls,
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            total_tokens=self.usage.total_tokens,
        )


@RemoteLanguageModel.register("alibaba")
class AlibabaModel(HttpServiceModel):
    _BASE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc"
    META = RemoteLanguageModelMetaInfo(
        language_models=[
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/api-details?spm=a2c4g.11186623.0.0.4070e0f6LPgURw#b8ebf6b25eul6
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen-max-0428",
            "qwen-max-0403",
            "qwen-max-0107",
            "qwen-max-longcontext",
            "qwen-turbo-online",
            "qwen-plus-online",
            "qwen-max-online",
            "qwen-max-0428-online",
            "qwen-max-0403-online",
            "qwen-max-0107-online",
            "qwen-max-longcontext-online",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/api-details-11?spm=a2c4g.11186623.0.0.7d4d23edoHHGiM#8f79b5d0f8ker
            "llama2-7b-chat-v2",
            "llama2-13b-chat-v2",
            "llama3-8b-instruct",
            "llama3-70b-instruct",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-7b-14b-72b-api-detailes?spm=a2c4g.11186623.0.0.34ad23edoCQT3i#8f79b5d0f8ker
            "qwen1.5-110b-chat",
            "qwen1.5-72b-chat",
            "qwen1.5-32b-chat",
            "qwen1.5-14b-chat",
            "qwen1.5-7b-chat",
            "qwen1.5-1.8b-chat",
            "qwen1.5-0.5b-chat",
            "codeqwen1.5-7b-chat",
            "qwen-72b-chat",
            "qwen-14b-chat",
            "qwen-7b-chat",
            "qwen-1.8b-longcontext-chat",
            "qwen-1.8b-chat",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/api-details-2?spm=a2c4g.11186623.0.0.14e09b6edreKfe#bd9128321bw93
            "baichuan-7b-v1",
            "baichuan2-7b-chat-v1",
            "baichuan2-13b-chat-v1",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/api-details-8?spm=a2c4g.11186623.0.0.4c565d88Iz6q95#bcdf80b31bxb8
            "chatglm-6b-v2",
            "chatglm3-6b",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/api-details-7?spm=a2c4g.11186623.0.0.3d04110fXv3GY6#8f79b5d0f8ker
            "moss-moon-003-sft-v1",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/api-details-3?spm=a2c4g.11186623.0.0.6f54312dwWsDWH#8f79b5d0f8ker
            "chatyuan-large-v2",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/yi-series-models-api-details?spm=a2c4g.11186623.0.0.64ef67e02E0ehT#bcdf80b31bxb8
            "yi-6b-chat",
            "yi-34b-chat",
            "aquilachat-7b",
            "deepseek-7b-chat",
        ],
        visual_language_models=[
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-plus-api?spm=a2c4g.11186623.0.0.391912b0mZS3gF
            "qwen-vl-plus",
            "qwen-vl-max",
            # noqa: https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-api?spm=a2c4g.11186623.0.0.62ec6dd76nY2Sp#8f79b5d0f8ker
            "qwen-vl-v1",
            "qwen-vl-chat-v1",
        ],
        tool_models=[
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen-max-0428",
            "qwen-max-0403",
            "qwen-max-0107",
            "qwen-max-longcontext",
        ],
        online_models=[
            "qwen-turbo-online",
            "qwen-plus-online",
            "qwen-max-online",
            "qwen-max-0428-online",
            "qwen-max-0403-online",
            "qwen-max-0107-online",
            "qwen-max-longcontext-online",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = AlibabaRequestBody
    RESPONSE_BODY_CLS = AlibabaResponseBody

    def _get_api_url(self):
        endpoint = "/text-generation/generation"
        if self.is_visual_model():
            endpoint = "/multimodal-generation/generation"

        return f"{self._BASE_URL}{endpoint}"

    def _make_api_headers(self):
        return {"Authorization": f"Bearer {self.config.api_key.get_secret_value()}"}

    @classmethod
    @validate_call
    def _convert_message(cls, message: ChatMessage) -> AlibabaChatMessage:
        return AlibabaChatMessage.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = [self._convert_message(message) for message in messages]
        if system:
            messages = [AlibabaChatMessage(role="system", content=system)] + messages

        if self.is_visual_model():
            for message in messages:
                if isinstance(message.content, str):
                    message.content = AlibabaContentPart(text=message.content)

        return {"input": {"messages": messages}}

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [OpenAITool.from_standard(tool) for tool in tools]

        return {"parameters": {"tools": tools}}

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "model": self.model.replace("-online", ""),
            "parameters": {
                "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
                "temperature": config.temperature or self.config.temperature,  # 默认 0.85
                "stop": config.stop_sequences or self.config.stop_sequences,
                "top_p": config.top_p or self.config.top_p,
                "top_k": config.top_k or self.config.top_k,  # >100 时 top_k 失效转而启用 top_p
                "repetition_penalty": config.repetition_penalty,  # 1.0 表示不做惩罚
                "enable_search": bool(self.is_online_model()),
            },
        }
