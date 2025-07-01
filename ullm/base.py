import mimetypes
import random
from abc import ABC, abstractclassmethod, abstractmethod
from copy import deepcopy
from itertools import chain
from operator import itemgetter
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

import jsonschema
import magic
import requests
from deepmerge import always_merger
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    Json,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    SecretStr,
    ValidationError,
    conlist,
    field_serializer,
    field_validator,
    model_validator,
    validate_call,
)


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    url: Optional[HttpUrl] = None
    path: Optional[str] = None
    mime_type: Optional[str] = None
    data: Optional[bytes] = None

    @model_validator(mode="before")
    @classmethod
    def check_image_data(cls, data):
        data = deepcopy(data)
        assert data.get("url") or data.get("path") or data.get("data")
        if data.get("path"):
            with open(data["path"], "rb") as image_file:
                data["data"] = image_file.read()

        if data.get("data"):
            data["mime_type"] = magic.from_buffer(data["data"], mime=True)
        else:
            data["mime_type"], _ = mimetypes.guess_type(data["url"])

        return data


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: conlist(Union[TextPart, ImagePart], min_length=1)

    @model_validator(mode="before")
    def validate_content(cls, data):
        data = deepcopy(data)
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            data["content"] = [TextPart(text=data["content"])]

        return data


class FunctionCall(BaseModel):
    name: str
    arguments: Optional[Union[Json, Dict[str, Any]]] = {}


class ToolCall(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: uuid4().hex)
    type: str
    function: Optional[FunctionCall] = None

    @model_validator(mode="after")
    def check_tool(self):
        if self.type == "function" and not self.function:
            raise ValueError("`function` should not be empty!")

        return self


class CitationSource(BaseModel):
    id: Optional[str] = None
    type: Literal["tool", "document"]
    tool_output: Optional[dict] = None
    document: Optional[dict] = None

    @model_validator(mode="before")
    def check_type(cls, values):
        if values["type"] == "tool":
            assert values.get("tool_output")

        if values["type"] == "document":
            assert values.get("document")

        return values


class Citation(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None
    text: Optional[str] = None
    sources: Optional[List[CitationSource]] = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = ""
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    citations: Optional[List[Citation]] = None

    @model_validator(mode="after")
    def check_content_or_tool_calls(self):
        assert self.content or self.tool_calls
        return self


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: Optional[str] = Field(default_factory=lambda: uuid4().hex)
    tool_name: str
    tool_result: str


ChatMessage = Union[UserMessage, AssistantMessage, ToolMessage]


class JsonSchemaObject(BaseModel):
    # https://spec.openapis.org/oas/v3.0.3#schema
    # https://ai.google.dev/api/rest/v1beta/Tool#Schema
    type: Literal["object"]
    properties: Optional[Dict[str, Any]] = {}
    required: Optional[List[str]] = []

    @model_validator(mode="before")
    @classmethod
    def validate_json_schema(cls, data):
        try:
            jsonschema.Draft7Validator.check_schema(data)
            return data
        except (ValueError, jsonschema.exceptions.SchemaError):
            raise ValueError("Not a valid json schema.")


class ParameterDefinition(BaseModel):
    type: str
    name: str
    description: str
    required: Optional[bool] = True


class FunctionObject(BaseModel):
    name: str
    description: str
    arguments: Optional[List[ParameterDefinition]] = []
    returns: Optional[List[ParameterDefinition]] = []


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionObject


class ToolChoice(BaseModel):
    # none: 不使用工具
    # auto: 模型自己判断
    # any: 从给定的工具中任选一个，如果不给定则从所有工具里选
    mode: Literal["none", "auto", "any"]
    functions: Optional[List[str]] = []


class GenerateConfig(BaseModel):
    temperature: Optional[NonNegativeFloat] = None
    max_tokens: Optional[PositiveInt] = None
    max_input_tokens: Optional[PositiveInt] = None
    max_output_tokens: Optional[PositiveInt] = None
    top_p: Optional[NonNegativeFloat] = None
    top_k: Optional[NonNegativeInt] = None
    stop_sequences: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    response_format: Optional[Literal["text", "json_object"]] = "text"
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    extra: Optional[dict] = None
    thinking_type: Optional[Literal["disabled", "enabled", "auto"]] = None


class GenerationResult(BaseModel):
    model: str
    stop_reason: str
    content: Optional[str] = ""
    reasoning_content: Optional[str] = ""
    tool_calls: Optional[List[ToolCall]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    original_result: Json[Any] = None
    citations: Optional[List[Citation]] = None

    @model_validator(mode="after")
    def check_content_or_tool_calls(self):
        assert self.content or self.tool_calls
        return self

    def to_message(self) -> AssistantMessage:
        return AssistantMessage(
            content=self.content, tool_calls=self.tool_calls, citations=self.citations
        )


class LanguageModel(ABC):
    def __init__(self, config):
        self.config = config

    @classmethod
    @validate_call
    def from_config(cls, config):
        if config["type"] == "local":
            return LocalLanguageModel._load(config)
        if config["type"] == "remote":
            return RemoteLanguageModel._load(config)

    @abstractclassmethod
    def _load(cls, config):
        pass

    @validate_call
    @abstractmethod
    def generate(self, prompt: str, config: Optional[GenerateConfig] = None) -> GenerationResult:
        pass

    @validate_call
    @abstractmethod
    def chat(
        self,
        messages: conlist(Union[UserMessage, AssistantMessage, ToolMessage], min_length=1),
        config: Optional[GenerateConfig] = None,
        system: Optional[str] = None,
    ) -> GenerationResult:
        pass

    def is_ready(self):
        return True


class ModelConfig(BaseModel):
    type: Literal["local", "remote"]


class LocalLanguageModelConfig(ModelConfig):
    """TBD"""


class LocalLanguageModel(LanguageModel):
    @classmethod
    def _load(cls, config: LocalLanguageModelConfig) -> "LocalLanguageModel":
        raise NotImplementedError


class RemoteLanguageModelConfig(ModelConfig):
    provider: str = Field(..., description="模型提供方名称")
    model: str = Field(..., description="模型名称")
    is_visual_model: Optional[bool] = Field(
        False,
        description="该模型是否支持视觉理解，用于自定义服务",
        examples=[True, False],
        json_schema_extra={"providers": ["openai-compatible"]},
    )
    is_tool_model: Optional[bool] = Field(
        False,
        description="是否支持工具，用于自定义服务",
        examples=[True, False],
        json_schema_extra={"providers": ["openai-compatible"]},
    )
    is_online_model: Optional[bool] = Field(
        False,
        description="是否支持联网，用于自定义服务",
        examples=[True, False],
        json_schema_extra={"providers": ["openai-compatible"]},
    )
    api_url: Optional[HttpUrl] = Field(
        None,
        description="有的 provider 并无公开的固定 URL 需要自己指定，如自己部署的 API 代理服务",
        examples=["http://example.com/api/v1/chat/completion"],
        json_schema_extra={"providers": ["openai-compatible"]},
    )
    api_key: Optional[SecretStr] = Field(
        "", examples=["sk-************************************************"]
    )
    secret_key: Optional[SecretStr] = Field(
        "",
        description="讯飞星火 api_secret, 文心一言 secret key，腾讯混元 secret_key",
        examples=["c5ff5142b0b248d5885bac25352364eb"],
        json_schema_extra={"providers": ["iflytek", "baidu"]},
    )
    azure_endpoint: Optional[str] = Field(
        "",
        description="用于 Azure OpenAI",
        examples=["https://example-endpoint.openai.azure.com/"],
        json_schema_extra={"providers": ["azure-openai"]},
    )
    azure_deployment_name: Optional[str] = Field(
        "",
        description="用于 Azure OpenAI",
        examples=["gpt-35-turbo"],
        json_schema_extra={"providers": ["azure-openai"]},
    )
    azure_api_version: Optional[str] = Field(
        "2024-02-01",
        description="用于 Azure OpenAI",
        examples=["2024-02-01"],
        json_schema_extra={"providers": ["azure-openai"]},
    )
    bytedance_endpoint: Optional[str] = Field(
        "",
        description="用于字节豆包模型",
        examples=["ep-20240101000000-abc123"],
        json_schema_extra={"providers": ["bytedance"]},
    )
    region: Optional[str] = Field(
        "",
        description="用于腾讯混元等需要指定地区的服务",
        examples=["ap-beijing"],
        json_schema_extra={"providers": ["tencent"]},
    )
    app_id: Optional[str] = Field(
        "",
        description="讯飞星火需要",
        examples=["404abcde"],
        json_schema_extra={"providers": ["iflytek"]},
    )
    max_tokens: Optional[PositiveInt] = Field(None, examples=[4096, 8192])
    max_input_tokens: Optional[PositiveInt] = Field(None, examples=[1024, 2048])
    max_output_tokens: Optional[PositiveInt] = Field(None, examples=[1024, 4096])
    temperature: Optional[NonNegativeFloat] = Field(None, examples=[0.7, 0.8])
    top_p: Optional[NonNegativeFloat] = Field(None, le=1.0, examples=[1.0])
    top_k: Optional[NonNegativeInt] = Field(None, examples=[50, 100])
    stop_sequences: Optional[List[str]] = Field(None, examples=[["stop1", "stop2"]])
    http_proxy: Optional[HttpUrl] = Field(None, examples=["https://example-proxy.com"])
    cf_account_id: Optional[SecretStr] = Field(
        None,
        description="Cloudflare Account ID",
        examples=["fe18f2a883e6401c9ee72ab358714088"],
        json_schema_extra={"providers": ["cloudflare"]},
    )

    @field_serializer("api_key", "secret_key", "cf_account_id", when_used="json")
    def dump_secret_json(self, secret):
        return secret.get_secret_value() if secret else None


class RemoteLanguageModelMetaInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    api_url: Optional[AnyUrl] = Field("", description="TODO")
    model_api_url_mappings: Optional[Dict[str, AnyUrl]] = Field({}, description="TODO")
    language_models: Optional[List[str]] = []
    visual_language_models: Optional[List[str]] = []
    tool_models: Optional[List[str]] = []
    online_models: Optional[List[str]] = []
    required_config_fields: Optional[List[str]] = []


class RemoteLanguageModel(LanguageModel):
    _PROVIDER_TO_SUB_CLS = {}
    META = RemoteLanguageModelMetaInfo()

    @property
    def model(self):
        return self.config.model

    def __repr__(self):
        return f"<RemoteModel: {self.config.provider} - {self.model}>"

    @classmethod
    def register(cls, provider):
        def wrap(sub_cls):
            if provider not in cls._PROVIDER_TO_SUB_CLS:
                cls._PROVIDER_TO_SUB_CLS[provider] = sub_cls

            return sub_cls

        return wrap

    def is_tool_model(self):
        if self.META.tool_models:
            return self.model in self.META.tool_models

        return self.config.is_tool_model

    def is_visual_model(self):
        if self.META.visual_language_models:
            return self.model in self.META.visual_language_models

        return self.config.is_visual_model

    def is_online_model(self):
        if self.META.online_models:
            return self.model in self.META.online_models

        return self.config.is_online_model

    def _validate_model(self, messages: conlist(ChatMessage, min_length=1)):
        message_parts = chain.from_iterable(
            [message.content for message in messages if isinstance(message, UserMessage)]
        )
        if any(isinstance(part, ImagePart) for part in message_parts):
            assert self.is_visual_model()

    @classmethod
    @validate_call
    def _get_supported_models(cls) -> conlist(str, min_length=1):
        return cls.META.language_models + cls.META.visual_language_models

    @classmethod
    def list_providers(cls):
        providers = []
        for provider, sub_cls in cls._PROVIDER_TO_SUB_CLS.items():
            models = sub_cls._get_supported_models()
            providers.append(
                {
                    "name": provider,
                    "models": len(models) or None,
                    "visual_models": len(sub_cls.META.visual_language_models) if models else None,
                    "tool_models": len(sub_cls.META.tool_models) if models else None,
                    "online_models": len(sub_cls.META.online_models) if models else None,
                }
            )

        return sorted(providers, key=itemgetter("name"))

    @classmethod
    def get_provider_example(cls, provider):
        sub_cls = cls._PROVIDER_TO_SUB_CLS.get(provider)
        if not sub_cls:
            return None

        meta = sub_cls.META
        example_model = meta.language_models[0] if meta.language_models else "model-name"
        required_config = {"type": "remote", "model": example_model, "provider": provider}
        for field in meta.required_config_fields:
            required_config[field] = random.choice(
                RemoteLanguageModelConfig.model_fields[field].examples
            )

        optional_config = {}
        for field, field_info in RemoteLanguageModelConfig.model_fields.items():
            if field in required_config:
                continue

            extra = field_info.json_schema_extra
            if extra and provider not in extra.get("providers", []):
                continue

            optional_config[field] = random.choice(field_info.examples)

        return required_config, optional_config

    @classmethod
    def list_models(
        cls,
        providers: Optional[List[str]] = None,
        visual: Optional[bool] = None,
        tools_enable: Optional[bool] = None,
        online: Optional[bool] = None,
    ):
        providers = providers or []
        models = []
        for provider, sub_cls in cls._PROVIDER_TO_SUB_CLS.items():
            if providers and provider not in providers:
                continue

            sub_cls_models = sub_cls._get_supported_models()
            for model in sub_cls_models:
                is_visual = model in sub_cls.META.visual_language_models
                if visual and not is_visual:
                    continue

                is_tool = model in sub_cls.META.tool_models
                if tools_enable and not is_tool:
                    continue

                is_online = model in sub_cls.META.online_models
                if online and not is_online:
                    continue

                models.append(
                    {
                        "provider": provider,
                        "model": model,
                        "visual": is_visual,
                        "online": is_online,
                        "tools": is_tool,
                    }
                )

        return sorted(models, key=itemgetter("provider", "model"))

    def __init__(self, config):
        super().__init__(config)
        self.validate_config()

    def validate_config(self):
        supported_models = self._get_supported_models()
        if supported_models and self.model not in self._get_supported_models():
            raise ValueError(f"Unsupported model: {self.model}")

        original_config = self.config.model_dump(exclude_unset=True)
        for field in self.META.required_config_fields:
            if field not in original_config:
                raise ValueError(f"config field missed: {field}")

    @classmethod
    @validate_call
    def _load(cls, config: RemoteLanguageModelConfig):
        sub_cls = cls._PROVIDER_TO_SUB_CLS.get(config.provider)
        if not sub_cls:
            raise ValueError(f"Unsupported provider: {config.provider}")

        return sub_cls(config)

    def _get_api_url(self):
        return (
            self.META.api_url or self.config.api_url or self.META.model_api_url_mappings[self.model]
        )

    @validate_call
    def generate(self, prompt: str, config: Optional[GenerateConfig] = None) -> GenerationResult:
        return self.chat([UserMessage(content=prompt)], config=config)

    def is_ready(self):
        ready = True
        try:
            response = self.generate("hello", config={"max_output_tokens": 1})
            if response.stop_reason == "error":
                ready = False
        except Exception:
            ready = False

        return ready


class HttpRequestData(BaseModel):
    headers: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    body: Optional[Dict[str, Any]] = None

    @field_validator("*")
    @classmethod
    def remove_none_value(cls, values):
        if not values:
            return values

        values = {key: value for key, value in values.items() if value is not None}
        return values


class HttpServiceModel(RemoteLanguageModel):
    REQUEST_BODY_CLS = None
    RESPONSE_BODY_CLS = None
    EXTRA_CONFIG_CLS = None

    @abstractmethod
    def _make_api_headers(self):
        pass

    def _make_api_params(self):
        """Google/Baidu 的接口需要 params，其他接口不需要，所以这个不作为 abstractmethod"""
        return None

    @abstractclassmethod
    def _convert_message(cls, message: ChatMessage):
        pass

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        # FIXME: model 不支持 tools 的时候，如果 messages 里有 ToolMessage
        # 和带 ToolCall 的 AssistantMessage 怎么办？
        return {"messages": [self._convert_message(message) for message in messages]}

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        return {"tools": None, "tool_choice": None}

    @validate_call
    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        return config.model_dump(exclude_none=True)

    @validate_call
    def _convert_extra_generation_config(
        self, extra_config: Optional[dict] = None
    ) -> Dict[str, Any]:
        if not extra_config:
            return {}

        if self.EXTRA_CONFIG_CLS:
            return self.EXTRA_CONFIG_CLS.model_validate(extra_config).model_dump(
                mode="json", exclude_none=True
            )

        return extra_config

    @validate_call
    def _make_api_body(
        self,
        messages: conlist(ChatMessage, min_length=1),
        config: GenerateConfig,
        system: Optional[str] = None,
    ):
        body = {}
        always_merger.merge(body, self._convert_messages(messages, system=system))
        if self.is_tool_model():
            always_merger.merge(body, self._convert_tools(config.tools, config.tool_choice))
        elif config.tools:
            pass

        always_merger.merge(body, self._convert_generation_config(config, system=system))
        always_merger.merge(body, self._convert_extra_generation_config(config.extra))
        return self.REQUEST_BODY_CLS.model_validate(body).model_dump(
            exclude_none=True, by_alias=True
        )

    @validate_call
    def chat(
        self,
        messages: conlist(ChatMessage, min_length=1),
        config: Optional[GenerateConfig] = None,
        system: Optional[str] = None,
    ) -> GenerationResult:
        config = config or GenerateConfig()
        self._validate_model(messages)
        api_url = self._get_api_url()
        request_data = HttpRequestData(
            headers=self._make_api_headers(),
            params=self._make_api_params(),
            body=self._make_api_body(messages, config, system=system),
        )
        return self._call_api(api_url, request_data)

    def _is_valid_response(cls, http_response):
        return http_response.status_code == 200

    def _parse_response(self, http_response) -> GenerationResult:
        return self.RESPONSE_BODY_CLS.model_validate(http_response.json()).to_standard(
            model=self.model
        )

    def _parse_error_response(self, http_response) -> GenerationResult:
        return GenerationResult(
            model=self.model,
            stop_reason="error",
            content=f"Error {http_response.status_code}: {http_response.text}",
        )

    @validate_call
    def _call_api(self, api_url: HttpUrl, request_data: HttpRequestData) -> GenerationResult:
        proxies = None
        if self.config.http_proxy:
            proxies = {"http": str(self.config.http_proxy), "https": str(self.config.http_proxy)}

        response = requests.post(
            api_url,
            params=request_data.params,
            headers=request_data.headers,
            json=request_data.body,
            proxies=proxies,
        )
        result = None
        if self._is_valid_response(response):
            try:
                result = self._parse_response(response)
            except ValidationError:
                result = self._parse_error_response(response)
        else:
            result = self._parse_error_response(response)

        result.original_result = response.text
        return result
