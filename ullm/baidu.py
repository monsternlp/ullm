import json
from typing import Any, Dict, List, Literal, Optional

import requests
from pydantic import (
    AnyUrl,
    BaseModel,
    PositiveInt,
    confloat,
    conint,
    conlist,
    model_validator,
    validate_call,
)

from .base import (
    AssistantMessage,
    ChatMessage,
    FunctionCall,
    FunctionObject,
    GenerateConfig,
    GenerationResult,
    HttpServiceModel,
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


class BaiduFunctionCall(BaseModel):
    name: str
    arguments: str
    thoughts: Optional[str] = None

    @classmethod
    def from_standard(cls, function_call):
        return cls(
            name=function_call.name,
            arguments=json.dumps(function_call.arguments, ensure_ascii=False),
        )


class BaiduChatMessage(BaseModel):
    name: Optional[str] = None
    role: Literal["user", "assistant", "function"]
    content: Optional[str] = None
    function_call: Optional[BaiduFunctionCall] = None

    @model_validator(mode="before")
    @classmethod
    def check_content_or_function(cls, values):
        assert values.get("content") or values.get("function_call")
        return values

    @classmethod
    def from_standard(cls, message: ChatMessage):
        if isinstance(message, UserMessage):
            content = ""
            for part in message.content:
                if isinstance(part, TextPart):
                    content += part.text + "\n"
                else:
                    pass

            return BaiduChatMessage(role="user", content=content.strip())
        if isinstance(message, AssistantMessage):
            if message.tool_calls and len(message.tool_calls) > 1:
                raise ValueError("`BaiduChatMessage` support single function call only.")

            function_call = None
            if message.tool_calls:
                function_call = BaiduFunctionCall.from_standard(message.tool_calls[0].function)

            return cls(role="assistant", content=message.content, function_call=function_call)
        if isinstance(message, ToolMessage):
            try:
                tool_result = json.loads(message.tool_result)
            except (TypeError, json.JSONDecodeError):
                tool_result = {"content": message.tool_result}

            return cls(
                name=message.tool_name,
                role="function",
                content=json.dumps(tool_result, ensure_ascii=False),
            )


class BaiduFunction(BaseModel):
    name: str
    description: str
    parameters: JsonSchemaObject
    responses: Optional[JsonSchemaObject] = None
    examples: Optional[List[BaiduChatMessage]] = None

    @classmethod
    def from_standard(cls, function: FunctionObject):
        data = {"name": function.name, "description": function.description}
        for name, value in zip(["parameters", "responses"], [function.arguments, function.returns]):
            properties, required = {}, []
            for parameter in value or []:
                if parameter.required:
                    required.append(parameter.name)

                properties[parameter.name] = {
                    "type": parameter.type,
                    "description": parameter.description,
                }

            data[name] = {"type": "object", "properties": properties, "required": required}

        return cls.model_validate(data)


class BaiduToolChoiceFunction(BaseModel):
    name: str


class BaiduToolChoice(BaseModel):
    type: Optional[Literal["function"]] = "function"
    function: BaiduToolChoiceFunction


class BaiduRequestBody(BaseModel):
    system: Optional[str] = None
    messages: conlist(BaiduChatMessage, min_length=1)
    temperature: Optional[confloat(gt=0, le=1.0)] = None
    top_p: Optional[confloat(ge=0, le=1.0)] = None
    top_k: Optional[PositiveInt] = None
    max_output_tokens: Optional[conint(ge=2, le=2048)] = None
    penalty_score: Optional[confloat(ge=1.0, le=2.0)] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = None

    # 只有 ERNIE-4.0 和 ERNIE-3.5 有这个参数
    response_format: Optional[Literal["text", "json_object"]] = None

    # 只有 ERNIE-4.0/ERNIE-3.5/ERNIE-Novel-8K/Qianfan-Dynamic-8K 有这三个参数
    disable_search: Optional[bool] = None
    enable_citation: Optional[bool] = None
    enable_trace: Optional[bool] = None

    # 只有 ERNIE-3.5 有这两个参数
    functions: Optional[List[BaiduFunction]] = None
    tool_choice: Optional[BaiduToolChoice] = None

    user_id: Optional[str] = None


class BaiduSearchResult(BaseModel):
    index: int
    url: AnyUrl
    title: str


class BaiduSearchInfo(BaseModel):
    search_results: Optional[List[BaiduSearchResult]] = None


class BaiduPluginUsage(BaseModel):
    name: str
    parse_tokens: Optional[int] = None
    abstract_tokens: Optional[int] = None
    search_tokens: Optional[int] = None
    total_tokens: int


class BaiduResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    plugins: Optional[List[BaiduPluginUsage]] = None


class BaiduResponseBody(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    result: Optional[str] = ""
    search_info: Optional[BaiduSearchInfo] = None
    usage: BaiduResponseUsage
    # NOTE: 仅 ERNIE-4.0 和 ERNIE-3.5 返回结果中有 finish_reason
    finish_reason: Optional[
        Literal["normal", "stop", "length", "content_filter", "function_call"]
    ] = "normal"
    is_truncated: Optional[bool] = None
    need_clear_history: Optional[bool] = None
    ban_round: Optional[int] = None
    flag: Optional[int] = None
    function_call: Optional[BaiduFunctionCall] = None

    def to_standard(self, model: str = None):
        tool_calls = None
        if self.function_call:
            tool_calls = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=self.function_call.name, arguments=self.function_call.arguments
                    ),
                )
            ]

        return GenerationResult(
            model=model,
            stop_reason=self.finish_reason,
            content=self.result,
            tool_calls=tool_calls,
            input_tokens=self.usage.prompt_tokens,
            output_tokens=self.usage.completion_tokens,
            total_tokens=self.usage.total_tokens,
        )


@RemoteLanguageModel.register("baidu")
class BaiduModel(HttpServiceModel):
    # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu
    # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/flxu4ej5u
    _BASE_URL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop"
    META = RemoteLanguageModelMetaInfo(
        model_api_url_mappings={
            "ERNIE-4.0-Turbo-128K": f"{_BASE_URL}/chat/ernie-4.0-turbo-128k",
            "ERNIE-4.0-Turbo-8K-Latest": f"{_BASE_URL}/chat/ernie-4.0-turbo-8k-latest",
            "ERNIE-4.0-Turbo-8K": f"{_BASE_URL}/chat/ernie-4.0-turbo-8k",
            "ERNIE-4.0-Turbo-8K-Preview": f"{_BASE_URL}/chat/ernie-4.0-turbo-8k-preview",
            "ERNIE-4.0-Turbo-8K-0628": f"{_BASE_URL}/chat/ernie-4.0-turbo-8k-0628",
            "ERNIE-4.0-8K": f"{_BASE_URL}/chat/completions_pro",
            "ERNIE-4.0-8K-Preview": f"{_BASE_URL}/chat/ernie-4.0-8k-preview",
            "ERNIE-4.0-8K-Latest": f"{_BASE_URL}/chat/ernie-4.0-8k-latest",
            "ERNIE-4.0-8K-0613": f"{_BASE_URL}/chat/ernie-4.0-8k-0613",
            "ERNIE-3.5-8K": f"{_BASE_URL}/chat/completions",
            "ERNIE-3.5-8K-0701": f"{_BASE_URL}/chat/ernie-3.5-8k-0701",
            "ERNIE-3.5-8K-0613": f"{_BASE_URL}/chat/ernie-3.5-8k-0613",
            "ERNIE-3.5-8K-Preview": f"{_BASE_URL}/chat/ernie-3.5-8k-preview",
            "ERNIE-3.5-128K": f"{_BASE_URL}/chat/ernie-3.5-128k",
            "ERNIE-3.5-128K-Preview": f"{_BASE_URL}/chat/ernie-3.5-128k-preview",
            "ERNIE-Speed-8K": f"{_BASE_URL}/chat/ernie_speed",
            "ERNIE-Speed-128K": f"{_BASE_URL}/chat/ernie-speed-128k",
            "ERNIE-Speed-Pro-128K": f"{_BASE_URL}/chat/ernie-speed-pro-128k",
            "ERNIE-Lite-8K": f"{_BASE_URL}/chat/ernie-lite-8k",
            "ERNIE-Lite-Pro-128K": f"{_BASE_URL}/chat/ernie-lite-pro-128k",
            # ERNIE-Lite-8K-0725/ERNIE-Lite-4K-0704/ERNIE-Lite-4K-0516/ERNIE-Lite-128K-0419
            # ERNIE-Tiny-128K-0929
            # 需要用户自己部署
            "ERNIE-Tiny-8K": f"{_BASE_URL}/chat/ernie-tiny-8k",
            "ERNIE-Character-8K-0321": f"{_BASE_URL}/chat/ernie-char-8k",
            "ERNIE-Functions-8K": f"{_BASE_URL}/chat/ernie-func-8k",
            "ERNIE-Novel-8K": f"{_BASE_URL}/chat/ernie-novel-8k",
            "Qianfan-Dynamic-8K": f"{_BASE_URL}/chat/qianfan-dynamic-8k",
            # ERNIE Speed-AppBuilder 需配合 AppBuilder-SDK 单独使用
            # Gemma-2B-it 需要用户自己部署
            "Gemma-7B-it": f"{_BASE_URL}/chat/gemma_7b_it",
            "Yi-34B-Chat": f"{_BASE_URL}/chat/yi_34b_chat",
            "Mixtral-8x7B-Instruct": f"{_BASE_URL}/chat/mixtral_8x7b_instruct",
            # Mistral-7B-Instruct 需要用户自己部署
            # Qianfan-Chinese-Llama-2-7B-32K, Qianfan-Chinese-Llama-2-13B-v2 需要用户自己部署
            "Qianfan-Chinese-Llama-2-7B": f"{_BASE_URL}/chat/qianfan_chinese_llama_2_7b",
            "Qianfan-Chinese-Llama-2-13B-v1": f"{_BASE_URL}/chat/qianfan_chinese_llama_2_13b",
            "Qianfan_Chinese_Llama_2-70B": f"{_BASE_URL}/chat/qianfan_chinese_llama_2_70b",
            # Qianfan-Chinese-Llama-2-70B/Qianfan-Chinese-Llama-2-1.3B 需要用户自己部署
            # Qianfan-Llama-2-70B-compressed 需要用户自己部署
            # Linly-Chinese-LLaMA-2-7B, Linly-Chinese-LLaMA-2-13B 需要用户自己部署
            "Llama-2-7b-chat": f"{_BASE_URL}/chat/llama_2_7b",
            "Llama-2-13b-chat": f"{_BASE_URL}/chat/llama_2_13b",
            "Llama-2-70b-chat": f"{_BASE_URL}/chat/llama_2_70b",
            "Meta-Llama-3-8B-Instruct": f"{_BASE_URL}/chat/llama_3_8b",
            "Meta-Llama-3-70B-Instruct": f"{_BASE_URL}/chat/llama_3_70b",
            # ChatGLM3-6B/chatglm3-6b-32k/ChatGLM2-6B-INT4/ChatGLM2-6B 需要用户自己部署
            "ChatGLM2-6B-32K": f"{_BASE_URL}/chat/chatglm2_6b_32k",
            # Baichuan2-7B-Chat/Baichuan2-13B-Chat 需要用户自己部署
            # XVERSE-13B-Chat 需要用户自己部署
            "XuanYuan-70B-Chat-4bit": f"{_BASE_URL}/chat/xuanyuan_70b_chat",
            # DISC-MedLLM 需要用户自己部署
            # NOTE: ChatLaw 测试返回 Error 336000: Internal error
            # "ChatLaw": f"{_BASE_URL}/chat/chatlaw",
            # Falcon-7B/Falcon-40B-Instruct 需要用户自己部署
            "AquilaChat-7B": f"{_BASE_URL}/chat/aquilachat_7b",
            # RWKV-4-World, RWKV-4-pile-14B, RWKV-Raven-14B 需要用户自己部署
            "BLOOMZ-7B": f"{_BASE_URL}/chat/bloomz_7b1",
            "Qianfan-BLOOMZ-7B-compressed": f"{_BASE_URL}/chat/qianfan_bloomz_7b_compressed",
            # OpenLLaMA-7B 需要用户自己部署
            # Dolly-12B 需要用户自己部署
            # MPT-7B-Instruct/MPT-30B-instruct 需要用户自己部署
            # OA-Pythia-12B-SFT-4 需要用户自己部署
        },
        language_models=[
            "ERNIE-4.0-Turbo-8K",
            "ERNIE-4.0-Turbo-8K-Preview",
            "ERNIE-4.0-Turbo-8K-0628",
            "ERNIE-4.0-Turbo-8K-online",
            "ERNIE-4.0-Turbo-8K-Preview-online",
            "ERNIE-4.0-Turbo-8K-0628-online",
            "ERNIE-4.0-8K",
            "ERNIE-4.0-8K-Preview",
            "ERNIE-4.0-8K-Latest",
            "ERNIE-4.0-8K-0613",
            "ERNIE-4.0-8K-online",
            "ERNIE-4.0-8K-Preview-online",
            "ERNIE-4.0-8K-Latest-online",
            "ERNIE-4.0-8K-0613-online",
            "ERNIE-3.5-8K",
            "ERNIE-3.5-8K-0701",
            "ERNIE-3.5-8K-0613",
            "ERNIE-3.5-8K-Preview",
            "ERNIE-3.5-128K",
            "ERNIE-3.5-8K-online",
            "ERNIE-3.5-8K-0701-online",
            "ERNIE-3.5-8K-0613-online",
            "ERNIE-3.5-8K-Preview-online",
            "ERNIE-3.5-128K-online",
            "ERNIE-Speed-8K",
            "ERNIE-Speed-128K",
            "ERNIE-Speed-Pro-128K",
            "ERNIE-Lite-8K",
            "ERNIE-Lite-Pro-128K",
            "ERNIE-Tiny-8K",
            "ERNIE-Character-8K-0321",
            "ERNIE-Functions-8K",
            "ERNIE-Novel-8K",
            "ERNIE-Novel-8K-online",
            "Qianfan-Dynamic-8K",
            "Qianfan-Dynamic-8K-online",
            "Gemma-7B-it",
            "Yi-34B-Chat",
            "Mixtral-8x7B-Instruct",
            "Qianfan-Chinese-Llama-2-7B",
            "Qianfan-Chinese-Llama-2-13B-v1",
            "Qianfan_Chinese_Llama_2-70B",
            "Llama-2-7b-chat",
            "Llama-2-13b-chat",
            "Llama-2-70b-chat",
            "Meta-Llama-3-8B-Instruct",
            "Meta-Llama-3-70B-Instruct",
            "ChatGLM2-6B-32K",
            "XuanYuan-70B-Chat-4bit",
            "AquilaChat-7B",
            "BLOOMZ-7B",
            "Qianfan-BLOOMZ-7B-compressed",
        ],
        visual_language_models=[],
        tool_models=[
            # NOTE: ERNIE-4.0 模型的 API 文档中没有写支持 functions 参数，
            #       但实际测试传入 functions 参数后会起作用
            "ERNIE-4.0-Turbo-8K",
            "ERNIE-4.0-Turbo-8K-Preview",
            "ERNIE-4.0-Turbo-8K-0628",
            "ERNIE-4.0-Turbo-8K-online",
            "ERNIE-4.0-Turbo-8K-Preview-online",
            "ERNIE-4.0-Turbo-8K-0628-online",
            "ERNIE-4.0-8K",
            "ERNIE-4.0-8K-Preview",
            "ERNIE-4.0-8K-Latest",
            "ERNIE-4.0-8K-0613",
            "ERNIE-4.0-8K-online",
            "ERNIE-4.0-8K-Preview-online",
            "ERNIE-4.0-8K-Latest-online",
            "ERNIE-4.0-8K-0613-online",
            "ERNIE-3.5-8K",
            "ERNIE-3.5-8K-0701",
            "ERNIE-3.5-8K-0613",
            "ERNIE-3.5-8K-Preview",
            "ERNIE-3.5-128K",
            "ERNIE-3.5-8K-online",
            "ERNIE-3.5-8K-0701-online",
            "ERNIE-3.5-8K-0613-online",
            "ERNIE-3.5-8K-Preview-online",
            "ERNIE-3.5-128K-online",
            # NOTE: ERNIE-Functions-8K 简介里说说适合外部工具使用，但文档中无 functions 参数，
            #       传入 functions 后也无作用
        ],
        online_models=[
            "ERNIE-4.0-Turbo-8K-online",
            "ERNIE-4.0-Turbo-8K-Preview-online",
            "ERNIE-4.0-Turbo-8K-0628-online",
            "ERNIE-4.0-8K-online",
            "ERNIE-4.0-8K-Preview-online",
            "ERNIE-4.0-8K-Latest-online",
            "ERNIE-4.0-8K-0613-online",
            "ERNIE-3.5-8K-online",
            "ERNIE-3.5-8K-0701-online",
            "ERNIE-3.5-8K-0613-online",
            "ERNIE-3.5-8K-Preview-online",
            "ERNIE-3.5-128K-online",
            "ERNIE-Novel-8K-online",
            "Qianfan-Dynamic-8K-online",
        ],
        required_config_fields=["api_key", "secret_key"],
    )
    REQUEST_BODY_CLS = BaiduRequestBody
    RESPONSE_BODY_CLS = BaiduResponseBody
    _AUTH_URL = "https://aip.baidubce.com/oauth/2.0/token"
    _MODEL_EXCLUDE_REQUEST_KEYS = {
        "ERNIE-4.0-Turbo-8K": ["top_k"],
        "ERNIE-4.0-Turbo-8K-Preview": ["top_k"],
        "ERNIE-4.0-Turbo-8K-0628": ["top_k"],
        "ERNIE-4.0-8K": ["top_k"],
        "ERNIE-4.0-8K-Preview": ["top_k"],
        "ERNIE-4.0-8K-Latest": ["top_k"],
        "ERNIE-4.0-8K-0613": ["top_k"],
        "ERNIE-3.5-8K": ["top_k"],
        "ERNIE-3.5-8K-0701": ["top_k"],
        "ERNIE-3.5-8K-0603": ["top_k"],
        "ERNIE-3.5-8K-Preview": ["top_k"],
        "ERNIE-3.5-128K": ["top_k"],
        "ERNIE-Speed-8K": [
            "top_k",
            "response_format",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "ERNIE-Speed-128K": [
            "top_k",
            "response_format",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "ERNIE-Speed-Pro-128K": [
            "top_k",
            "response_format",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "ERNIE-Lite-8K": [
            "top_k",
            "response_format",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "ERNIE-Lite-Pro-128K": [
            "top_k",
            "response_format",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "ERNIE-Tiny-8K": [
            "top_k",
            "response_format",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "ERNIE-Character-8K-0321": [
            "top_k",
            "response_format",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "ERNIE-Functions-8K": [
            "top_k",
            "response_format",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "ERNIE-Novel-8K": ["top_k"],
        "Qianfan-Dynamic-8K": ["top_k"],
        "Gemma-7B-it": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "Yi-34B-Chat": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "Mixtral-8x7B-Instruct": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "Qianfan-Chinese-Llama-2-7B": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "Llama-2-7b-chat": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "Llama-2-13b-chat": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "Qianfan-Chinese-Llama-2-13B-v1": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "Qianfan-Chinese-Llama-2-70B": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "Llama-2-70b-chat": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "Meta-Llama-3-8B-Instruct": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "Meta-Llama-3-70B-Instruct": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "ChatGLM2-6B-32K": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "XuanYuan-70B-Chat-4bit": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "AquilaChat-7B": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "BLOOMZ-7B": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
        "Qianfan-BLOOMZ-7B-compressed": [
            "response_format",
            "max_output_tokens",
            "system",
            "disable_search",
            "enable_citation",
            "enable_trace",
        ],
    }

    def _get_api_url(self):
        return self.META.model_api_url_mappings[self.model.replace("-online", "")]

    def get_access_token(self):
        payload = json.dumps("")
        params = {
            "grant_type": "client_credentials",
            "client_id": self.config.api_key.get_secret_value(),
            "client_secret": self.config.secret_key.get_secret_value(),
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(self._AUTH_URL, headers=headers, params=params, data=payload)
        return response.json().get("access_token")

    def _make_api_headers(self):
        return None

    def _make_api_params(self):
        access_token = self.get_access_token()
        return {"access_token": access_token}

    @classmethod
    def _convert_message(cls, message: ChatMessage):
        return BaiduChatMessage.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = {"messages": [self._convert_message(message) for message in messages]}
        if system:
            result["system"] = system

        return result

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if not self.is_tool_model or (tool_choice and tool_choice.mode == "none"):
            return {}

        data = {}
        if tools:
            data["functions"] = [BaiduFunction.from_standard(tool.function) for tool in tools]

        if not tool_choice or tool_choice.mode == "auto" or len(tool_choice.functions or []) != 1:
            return data

        data["tool_choice"] = {"type": "function", "function": {"name": tool_choice.functions[0]}}
        return data

    @validate_call
    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        stop = config.stop_sequences or self.config.stop_sequences
        if isinstance(stop, str):
            stop = [stop]

        data = {
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
            "top_k": config.top_k or self.config.top_k,
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "stop": stop,
            # "penalty_score": None
            "response_format": config.response_format,
        }
        if self.is_online_model():
            data["disable_search"] = False
            data["enable_trace"] = True
        else:
            data["disable_search"] = True
            data["enable_trace"] = False

        exclude_keys = self._MODEL_EXCLUDE_REQUEST_KEYS.get(self.model.replace("-online", ""), [])
        if exclude_keys:
            data = {key: value for key, value in data.items() if key not in exclude_keys}

        return data

    def _is_valid_response(cls, http_response):
        return http_response.status_code == 200 and http_response.text.find("error_code") < 0

    def _parse_error_response(self, http_response) -> GenerationResult:
        error_code = http_response.status_code
        error_message = http_response.text
        if http_response.text.find("error_code") > 0:
            response = http_response.json()
            error_code = response["error_code"]
            error_message = response["error_msg"]

        return GenerationResult(
            model=self.model,
            stop_reason="error",
            content=f"Error {error_code}: {error_message}",
        )
