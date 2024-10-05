from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    validate_call,
)

from .base import (
    GenerateConfig,
    GenerationResult,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    Tool,
    ToolCall,
    ToolChoice,
)
from .openai import (
    OpenAIAssistantMessage,
    OpenAICompatibleModel,
    OpenAIFunctionObject,
    OpenAIRequestBody,
    OpenAIResponseUsage,
    OpenAIToolChoice,
)


class IflyTekWebSearch(BaseModel):
    enable: Optional[bool] = None


class IflyTekTool(BaseModel):
    type: Literal["function", "web_search"]
    function: Optional[OpenAIFunctionObject] = None
    web_search: Optional[IflyTekWebSearch] = None

    @classmethod
    def from_standard(cls, tool: Tool):
        return cls(type=tool.type, function=OpenAIFunctionObject.from_standard(tool.function))


class IflyTekRequestBody(OpenAIRequestBody):
    # Reference: https://www.xfyun.cn/doc/spark//HTTP调用文档.html
    ## exclude fields
    frequency_penalty: Optional[Any] = Field(None, exclude=True)
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(None, exclude=True)
    presence_penalty: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    stop: Optional[Any] = Field(None, exclude=True)
    top_p: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)

    ## different parameters
    tools: Optional[List[IflyTekTool]] = None

    ## IflyTek-specific parameters
    top_k: Optional[int] = None


class IflyTekAssistantMessage(OpenAIAssistantMessage):
    @model_validator(mode="before")
    def check_tool_calls(cls, values):
        if values.get("tool_calls") and not isinstance(values["tool_calls"], list):
            values["tool_calls"] = [values["tool_calls"]]

        return values


class IflyTekResponseChoice(BaseModel):
    index: int
    message: IflyTekAssistantMessage


class IflyTekResponseBody(BaseModel):
    # https://www.xfyun.cn/doc/spark/Web.html#_1-%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E
    code: int
    message: str
    sid: str
    choices: List[IflyTekResponseChoice]
    usage: Optional[OpenAIResponseUsage] = None

    def to_standard(self, model: str):
        message = self.choices[0].message
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                tool_calls.append(ToolCall.model_validate(tool_call.model_dump()))

        return GenerationResult(
            model=model,
            stop_reason="stop",
            content=message.content,
            tool_calls=tool_calls,
            input_tokens=self.usage.prompt_tokens,
            output_tokens=self.usage.completion_tokens,
            total_tokens=self.usage.total_tokens,
        )


@RemoteLanguageModel.register("iflytek")
class IflyTekModel(OpenAICompatibleModel):
    META = RemoteLanguageModelMetaInfo(
        api_url="https://spark-api-open.xf-yun.com/v1/chat/completions",
        language_models=[
            "SparkDesk-4.0-Ultra",
            "SparkDesk-Max-32K",
            "SparkDesk-Max",
            "SparkDesk-Pro-128K",
            "SparkDesk-Pro",
            "SparkDesk-Lite",
            "SparkDesk-4.0-Ultra-online",
            "SparkDesk-Max-32K-online",
            "SparkDesk-Max-online",
            "SparkDesk-Pro-128K-online",
            "SparkDesk-Pro-online",
            "SparkDesk-Lite-online",
        ],
        visual_language_models=[],
        tool_models=[
            "SparkDesk-4.0-Ultra",
            "SparkDesk-Max-32K",
            "SparkDesk-Max",
            "SparkDesk-Pro-128K",
            "SparkDesk-Pro",
        ],
        online_models=[
            "SparkDesk-4.0-Ultra-online",
            "SparkDesk-Max-32K-online",
            "SparkDesk-Max-online",
            "SparkDesk-Pro-128K-online",
            "SparkDesk-Pro-online",
            "SparkDesk-Lite-online",
        ],
        required_config_fields=["api_key"],
    )
    _MODEL_MAPPINGS = {
        "SparkDesk-4.0-Ultra": "4.0Ultra",
        "SparkDesk-Max-32K": "max-32k",
        "SparkDesk-Max": "generalv3.5",
        "SparkDesk-Pro-128K": "pro-128k",
        "SparkDesk-Pro": "generalv3",
        "SparkDesk-Lite": "general",
    }
    REQUEST_BODY_CLS = IflyTekRequestBody
    RESPONSE_BODY_CLS = IflyTekResponseBody

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [IflyTekTool.from_standard(tool) for tool in tools]

        if self.is_online_model():
            tools = tools or []
            tools.append({"type": "web_search", "web_search": {"enable": True}})
        else:
            tools = tools or []
            tools.append({"type": "web_search", "web_search": {"enable": False}})

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
            "model": self._MODEL_MAPPINGS[self.model.replace("-online", "")],
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "temperature": config.temperature or self.config.temperature,
            "top_k": config.top_k or self.config.top_k,
        }
