import random
from abc import ABC, abstractclassmethod, abstractmethod
from itertools import chain
from operator import itemgetter
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Optional, Union

import requests
from deepmerge import always_merger
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    SecretStr,
    ValidationError,
    field_serializer,
    field_validator,
    validate_call,
)

from .types import (
    AssistantMessage,
    ChatMessage,
    GenerateConfig,
    GenerationResult,
    ImagePart,
    TextPart,
    Tool,
    ToolChoice,
    UserMessage,
)

if TYPE_CHECKING:
    pass


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
        messages: Annotated[List[ChatMessage], Field(min_length=1)],
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
        default=False,
        description="该模型是否支持视觉理解，用于自定义服务",
        examples=[True, False],
        json_schema_extra={"providers": ["openai-compatible"]},
    )
    is_tool_model: Optional[bool] = Field(
        default=False,
        description="是否支持工具，用于自定义服务",
        examples=[True, False],
        json_schema_extra={"providers": ["openai-compatible"]},
    )
    is_online_model: Optional[bool] = Field(
        default=False,
        description="是否支持联网，用于自定义服务",
        examples=[True, False],
        json_schema_extra={"providers": ["openai-compatible"]},
    )
    api_url: Optional[HttpUrl] = Field(
        default=None,
        description="有的 provider 并无公开的固定 URL 需要自己指定，如自己部署的 API 代理服务",
        examples=["http://example.com/api/v1/chat/completion"],
        json_schema_extra={"providers": ["openai-compatible"]},
    )
    api_key: Optional[SecretStr] = Field(
        default=SecretStr(""), examples=["sk-************************************************"]
    )
    secret_key: Optional[SecretStr] = Field(
        default=SecretStr(""),
        description="讯飞星火 api_secret, 文心一言 secret key，腾讯混元 secret_key",
        examples=["c5ff5142b0b248d5885bac25352364eb"],
        json_schema_extra={"providers": ["iflytek", "baidu"]},
    )
    azure_endpoint: Optional[str] = Field(
        default="",
        description="用于 Azure OpenAI",
        examples=["https://example-endpoint.openai.azure.com/"],
        json_schema_extra={"providers": ["azure-openai"]},
    )
    azure_deployment_name: Optional[str] = Field(
        default="",
        description="用于 Azure OpenAI",
        examples=["gpt-35-turbo"],
        json_schema_extra={"providers": ["azure-openai"]},
    )
    azure_api_version: Optional[str] = Field(
        default="2024-02-01",
        description="用于 Azure OpenAI",
        examples=["2024-02-01"],
        json_schema_extra={"providers": ["azure-openai"]},
    )
    bytedance_endpoint: Optional[str] = Field(
        default="",
        description="用于字节豆包模型",
        examples=["ep-20240101000000-abc123"],
        json_schema_extra={"providers": ["bytedance"]},
    )
    region: Optional[str] = Field(
        default="",
        description="用于腾讯混元等需要指定地区的服务",
        examples=["ap-beijing"],
        json_schema_extra={"providers": ["tencent"]},
    )
    app_id: Optional[str] = Field(
        default="",
        description="讯飞星火需要",
        examples=["404abcde"],
        json_schema_extra={"providers": ["iflytek"]},
    )
    max_tokens: Optional[PositiveInt] = Field(default=None, examples=[4096, 8192])
    max_input_tokens: Optional[PositiveInt] = Field(default=None, examples=[1024, 2048])
    max_output_tokens: Optional[PositiveInt] = Field(default=None, examples=[1024, 4096])
    temperature: Optional[NonNegativeFloat] = Field(default=None, examples=[0.7, 0.8])
    top_p: Optional[NonNegativeFloat] = Field(default=None, le=1.0, examples=[1.0])
    top_k: Optional[NonNegativeInt] = Field(default=None, examples=[50, 100])
    stop_sequences: Optional[List[str]] = Field(default=None, examples=[["stop1", "stop2"]])
    http_proxy: Optional[HttpUrl] = Field(default=None, examples=["https://example-proxy.com"])
    cf_account_id: Optional[SecretStr] = Field(
        default=None,
        description="Cloudflare Account ID",
        examples=["fe18f2a883e6401c9ee72ab358714088"],
        json_schema_extra={"providers": ["cloudflare"]},
    )

    @field_serializer("api_key", "secret_key", "cf_account_id", when_used="json")
    def dump_secret_json(self, secret):
        return secret.get_secret_value() if secret else None


class RemoteLanguageModelMetaInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    api_url: Optional[AnyUrl] = Field(default=None, description="TODO")
    model_api_url_mappings: Optional[Dict[str, AnyUrl]] = Field(default={}, description="TODO")
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

    def _validate_model(self, messages: Annotated[List[ChatMessage], Field(min_length=1)]):
        message_parts = chain.from_iterable(
            [message.content for message in messages if isinstance(message, UserMessage)]
        )
        if any(isinstance(part, ImagePart) for part in message_parts):
            assert self.is_visual_model()

    @classmethod
    @validate_call
    def _get_supported_models(cls) -> Annotated[List[str], Field(min_length=1)]:
        return (cls.META.language_models or []) + (cls.META.visual_language_models or [])

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
        for field in self.META.required_config_fields or []:
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
            self.META.api_url
            or self.config.api_url
            or (self.META.model_api_url_mappings or {}).get(self.model)
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
        messages: Annotated[List[ChatMessage], Field(min_length=1)],
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
        messages: Annotated[List[ChatMessage], Field(min_length=1)],
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
        return self.REQUEST_BODY_CLS.model_validate(body).model_dump(  # type: ignore
            exclude_none=True, by_alias=True
        )

    @validate_call
    def chat(
        self,
        messages: Annotated[List[ChatMessage], Field(min_length=1)],
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

    def _is_valid_response(self, http_response):
        return http_response.status_code == 200

    def _parse_response(self, http_response) -> GenerationResult:
        if not self.RESPONSE_BODY_CLS:
            raise ValueError("RESPONSE_BODY_CLS is not set for this service model")
        return self.RESPONSE_BODY_CLS.model_validate(http_response.json()).to_standard(
            model=self.model
        )

    def _parse_error_response(self, http_response) -> GenerationResult:
        return GenerationResult(
            model=self.model,
            stop_reason="error",
            message=AssistantMessage(
                content=[TextPart(text=f"Error {http_response.status_code}: {http_response.text}")]
            ),
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

    @validate_call
    def chat_as_openai(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Dict[Literal["type"], Literal["text", "json_object"]]] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        user: Optional[str] = None,
    ):
        from .openai_types import OpenAIRequestBody, OpenAIResponseBody

        request_body = OpenAIRequestBody(
            messages=messages,
            model=model,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            user=user,
        )

        standard_request = request_body.to_standard()
        result = self.chat(
            messages=standard_request["messages"],
            config=standard_request["config"],
            system=standard_request["system"],
        )
        return OpenAIResponseBody.from_standard(result)
