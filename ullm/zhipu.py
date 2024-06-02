from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    confloat,
    model_validator,
    validate_call,
)

from .base import (
    GenerateConfig,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    Tool,
    ToolChoice,
)
from .openai import (
    OpenAICompatibleModel,
    OpenAIFunctionObject,
    OpenAIRequestBody,
    OpenAIResponseBody,
)


class ZhipuRetrievalTool(BaseModel):
    knowledge_id: str
    prompt_template: Optional[str] = None


class ZhipuWebSearchTool(BaseModel):
    enable: Optional[bool] = False
    search_query: Optional[str] = None


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
            "glm-3-turbo",
            "glm-3-turbo-online",
            "glm-4",
            "glm-4-online",
        ],
        visual_language_models=["glm-4v"],
        tool_models=[
            "glm-3-turbo",
            "glm-3-turbo-online",
            "glm-4",
            "glm-4-online",
        ],
        online_models=[
            "glm-3-turbo-online",
            "glm-4-online",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = ZhipuAIRequestBody
    RESPONSE_BODY_CLS = ZhipuAIResponseBody

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [ZhipuAITool.from_standard(tool) for tool in tools]

        if self.is_online_model():
            tools = tools or []
            tools.append(ZhipuAITool(type="web_search", web_search=ZhipuWebSearchTool(enable=True)))
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
