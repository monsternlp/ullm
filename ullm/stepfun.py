from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validate_call

from .base import (
    FunctionObject,
    GenerateConfig,
    JsonSchemaObject,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    Tool,
    ToolChoice,
)
from .openai import OpenAICompatibleModel, OpenAIRequestBody


class StepFunFunctionObject(BaseModel):
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


class StepFunTool(BaseModel):
    type: Literal["function", "web_search"]
    function: StepFunFunctionObject

    @classmethod
    def from_standard(cls, tool: Tool):
        return cls(type=tool.type, function=StepFunFunctionObject.from_standard(tool.function))


class StepFunRequestBody(OpenAIRequestBody):
    # reference: https://platform.stepfun.com/docs/Chat/chat-completion-create
    ## excluded parameters
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)

    ## different parameters
    tools: Optional[List[StepFunTool]] = None


@RemoteLanguageModel.register("stepfun")
class StepFunModel(OpenAICompatibleModel):
    # reference: https://platform.openai.com/docs/models/overview
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.stepfun.com/v1/chat/completions",
        language_models=[
            "step-1-8k",
            "step-1-32k",
            "step-1-128k",
            "step-1-256k",
            "step-1-flash",
            "step-1-8k-online",
            "step-1-32k-online",
            "step-1-128k-online",
            "step-1-256k-online",
            "step-1-flash-online",
            "step-1x-medium",
            "step-1-flash",
            "step-2-16k",
            "step-2-16k-online",
            "step-2-16k-exp",
            "step-2-mini",
        ],
        visual_language_models=[
            "step-1v-8k",
            "step-1v-32k",
            "step-1v-8k-online",
            "step-1v-32k-online",
            "step-1o-vision-32k",
            "step-1o-turbo-vision",
            "step-1.5v-mini",
        ],
        # https://platform.stepfun.com/docs/llm/modeloverview
        tool_models=[
            "step-1-8k",
            "step-1-32k",
            "step-1-128k",
            "step-1-256k",
            "step-1-flash",
            "step-1-8k-online",
            "step-1-32k-online",
            "step-1-128k-online",
            "step-1-256k-online",
            "step-1-flash-online",
            "step-1v-8k",
            "step-1v-32k",
            "step-1v-8k-online",
            "step-1v-32k-online",
            "step-2-16k",
            "step-2-16k-online",
        ],
        online_models=[
            "step-1-8k-online",
            "step-1-32k-online",
            "step-1-128k-online",
            "step-1-256k-online",
            "step-1-flash-online",
            "step-1v-8k-online",
            "step-1v-32k-online",
            "step-2-16k-online",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = StepFunRequestBody

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [StepFunTool.from_standard(tool) for tool in tools]

        return {"tools": tools}

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        generation_config = {
            "model": self.model.replace("-online", ""),
            "frequency_penalty": config.frequency_penalty,
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "presence_penalty": config.presence_penalty,
            "response_format": {"type": config.response_format} if config.response_format else None,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
        }
        if self.is_online_model():
            generation_config["tools"] = [
                {
                    "type": "web_search",
                    "function": {
                        "name": "web_search",
                        "description": "这个web_search用来搜索互联网的信息",
                    },
                }
            ]

        return generation_config
