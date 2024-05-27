import json
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, confloat, model_validator

from .base import (
    FunctionObject,
    GenerationResult,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    Tool,
)
from .openai import (
    OpenAICompatibleModel,
    OpenAIRequestBody,
    OpenAIResponseBody,
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
    type: Literal["function"]
    function: MiniMaxFunctionObject

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

    @model_validator(mode="before")
    @classmethod
    def check_tool_choice(cls, values):
        tool_choice = values.get("tool_choice")
        if tool_choice and not isinstance(tool_choice, str):
            values["tool_choice"] = None

        return values


class MiniMaxResponseUsage(BaseModel):
    total_tokens: int


class MiniMaxResponseBody(OpenAIResponseBody):
    usage: Optional[MiniMaxResponseUsage] = None

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
            "abab6.5-chat",
            "abab6.5s-chat",
            "abab6.5g-chat",
            "abab6-chat",
            "abab5.5-chat",
            "abab5.5s-chat",
        ],
        visual_language_models=[],
        tool_models=[
            "abab6.5-chat",
            "abab6.5s-chat",
            "abab6.5g-chat",
            "abab6-chat",
            "abab5.5-chat",
            "abab5.5s-chat",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = MiniMaxRequestBody
    RESPONSE_BODY_CLS = MiniMaxResponseBody
