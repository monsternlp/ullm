import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, confloat, conlist, model_validator, validate_call

from .base import (
    FunctionObject,
    GenerateConfig,
    GenerationResult,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    Tool,
    ToolChoice,
)
from .openai import (
    OpenAIAssistantMessage,
    OpenAICompatibleModel,
    OpenAIRequestBody,
    OpenAIResponseBody,
    OpenAIResponseChoice,
)


class MiniMaxFunctionObject(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[str] = None

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

        if parameters:
            return cls(
                name=function.name,
                description=function.description,
                parameters=json.dumps(
                    {"type": "object", "properties": parameters, "required": required}
                ),
            )

            return cls(
                name=function.name,
                description=function.description,
            )


class MiniMaxTool(BaseModel):
    type: Literal["function", "web_search"]
    function: Optional[MiniMaxFunctionObject] = None

    @classmethod
    def from_standard(cls, tool: Tool):
        return cls(type=tool.type, function=MiniMaxFunctionObject.from_standard(tool.function))


class MiniMaxRequestBody(OpenAIRequestBody):
    # https://www.minimaxi.com/document/guides/chat-model/V2?id=65e0736ab2845de20908e2dd
    ## excluded parameters
    frequency_penalty: Optional[Any] = Field(None, exclude=True)
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(None, exclude=True)
    presence_penalty: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)

    ## different parameters
    temperature: Optional[confloat(gt=0.0, le=1.0)] = None
    top_p: Optional[confloat(gt=0.0, le=1.0)] = None
    tools: Optional[List[MiniMaxTool]] = None
    tool_choice: Optional[Literal["auto", "none"]] = None

    ## MiniMax-specific parameters
    mask_sensitive_info: Optional[bool] = None


class MiniMaxResponseUsage(BaseModel):
    total_tokens: int


class MiniMaxBaseResponse(BaseModel):
    status_code: int
    status_msg: str


class MiniMaxResponseChoice(OpenAIResponseChoice):
    finish_reason: str
    index: int
    message: Optional[OpenAIAssistantMessage] = None
    messages: Optional[list] = None

    @model_validator(mode="before")
    @classmethod
    def parse_messages(cls, values):
        if "message" not in values and "messages" not in values:
            raise ValueError

        if "messages" in values:
            values["message"] = values["messages"][3]

        return values


class MiniMaxResponseBody(OpenAIResponseBody):
    choices: conlist(MiniMaxResponseChoice, min_length=1)
    usage: Optional[MiniMaxResponseUsage] = None
    input_sensitive: bool
    input_sensitive_type: Optional[int] = Field(
        None,
        description=(
            "当input_sensitive为true时返回。取值为以下其一："
            "1 严重违规；2 色情；3 广告；4 违禁；5 谩骂；6 暴恐；7 其他。"
        ),
    )
    output_sensitive: bool
    output_sensitive_type: Optional[int] = Field(
        None,
        description=(
            "当output_sensitive为true时返回。取值为以下其一："
            "1 严重违规；2 色情；3 广告；4 违禁；5 谩骂；6 暴恐；7 其他。"
        ),
    )
    base_resp: MiniMaxBaseResponse

    def to_standard(self, model: str = None):
        return GenerationResult(
            model=model or self.model,
            stop_reason=self.choices[0].finish_reason,
            content=self.choices[0].message.content,
            tool_calls=self.choices[0].message.tool_calls,
            total_tokens=self.usage.total_tokens,
        )


@RemoteLanguageModel.register("minimax")
class MiniMaxModel(OpenAICompatibleModel):
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.minimax.chat/v1/text/chatcompletion_v2",
        language_models=[
            "MiniMax-Text-01",
            "MiniMax-Text-01-online",
            "abab6.5s-chat",
            "abab6.5s-chat-online",
            "DeepSeek-R1",
            "DeepSeek-R1-online",
        ],
        visual_language_models=[],
        tool_models=[
            "MiniMax-Text-01",
            "MiniMax-Text-01-online",
            "abab6.5s-chat",
            "abab6.5s-chat-online",
            "DeepSeek-R1",
            "DeepSeek-R1-online",
        ],
        online_models=[
            "MiniMax-Text-01-online",
            "abab6.5s-chat-online",
            "DeepSeek-R1-online",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = MiniMaxRequestBody
    RESPONSE_BODY_CLS = MiniMaxResponseBody

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [MiniMaxTool.from_standard(tool) for tool in tools]

        if self.is_online_model():
            tools = tools or []
            tools.append(MiniMaxTool(type="web_search"))

        minimax_tool_choice = None
        if tools:
            minimax_tool_choice = None
            if tools and tool_choice is not None:
                minimax_tool_choice = tool_choice.mode
                if minimax_tool_choice == "any":
                    minimax_tool_choice = "auto"

        return {"tools": tools, "tool_choice": minimax_tool_choice}

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
