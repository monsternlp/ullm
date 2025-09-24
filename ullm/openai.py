from typing import Any, Dict, List, Optional

from pydantic import validate_call

from .base import (
    HttpServiceModel,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai_types import (
    AzureOpenAIRequestBody,
    OpenAIAssistantMessage,
    OpenAIChatMessage,
    OpenAIRequestBody,
    OpenAIResponseBody,
    OpenAISystemMessage,
    OpenAITool,
    OpenAIToolChoice,
    OpenAIToolMessage,
    OpenAIUserMessage,
    OpenRouterReasoning,
)
from .types import (
    AssistantMessage,
    ChatMessage,
    GenerateConfig,
    Tool,
    ToolChoice,
    ToolMessage,
    UserMessage,
)

# === OpenAI model implementations ===


@RemoteLanguageModel.register("openai-compatible")
class OpenAICompatibleModel(HttpServiceModel):
    META = RemoteLanguageModelMetaInfo(
        required_config_fields=["api_url", "is_visual_model", "is_tool_model"]
    )
    REQUEST_BODY_CLS = OpenAIRequestBody
    RESPONSE_BODY_CLS = OpenAIResponseBody

    def _make_api_headers(self):
        if self.config.api_key:
            return {"Authorization": f"Bearer {self.config.api_key.get_secret_value()}"}

        return None

    @classmethod
    @validate_call
    def _convert_message(cls, message: ChatMessage) -> OpenAIChatMessage:
        if isinstance(message, AssistantMessage):
            return OpenAIAssistantMessage.from_standard(message)
        elif isinstance(message, UserMessage):
            return OpenAIUserMessage.from_standard(message)
        elif isinstance(message, ToolMessage):
            return OpenAIToolMessage.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: List[ChatMessage],
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = [self._convert_message(message) for message in messages]
        if system:
            messages = [OpenAISystemMessage(content=system)] + messages

        return {"messages": messages}

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [OpenAITool.from_standard(tool) for tool in tools]

        openai_tool_choice = None
        if tools and tool_choice is not None:
            openai_tool_choice = tool_choice.mode
            if openai_tool_choice == "any":
                if tool_choice.functions and len(tool_choice.functions) == 1:
                    openai_tool_choice = OpenAIToolChoice(
                        type="function", function={"name": tool_choice.functions[0]}
                    )
                else:
                    raise ValueError("OpenAI does not supported multi functions in `tool_choice`")

        return {"tools": tools, "tool_choice": openai_tool_choice}

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        convert GenerateConfig to supported params in OpenAIRequestBody
        """
        params: Dict[str, Any] = {
            "model": self.model,
            "frequency_penalty": config.frequency_penalty,
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "presence_penalty": config.presence_penalty,
            "response_format": {"type": config.response_format} if config.response_format else None,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
            "modalities": config.modalities,
        }

        if config.thinking:
            params["reasoning"] = OpenRouterReasoning.from_standard(config.thinking)
        return params


@RemoteLanguageModel.register("openai")
class OpenAIModel(OpenAICompatibleModel):
    # reference: https://platform.openai.com/docs/models
    # TODO: gpt-4o-audio-preview/gpt-4o-audio-preview-2024-10-01
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.openai.com/v1/chat/completions",
        language_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-instruct",
            "gpt-3.5-turbo-instruct-0914",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-32k-0314",
            "o1-mini",
            "o1-mini-2024-09-12",
            "o1-preview",
            "o1-preview-2024-09-12",
            "o3-mini",
            "o3-mini-2025-01-31",
        ],
        visual_language_models=[
            "gpt-4-vision-preview",
            "gpt-4-0215-preview",
            "gpt-4-1106-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "chatgpt-4o-latest",
            "o1",
            "o1-2024-12-17",
        ],
        # https://platform.openai.com/docs/guides/function-calling
        tool_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-vision-preview",
            "gpt-4-0215-preview",
            "gpt-4-1106-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "o1",
            "o1-2024-12-17",
            "o1-preview",
            "o1-preview-2024-09-12",
            "o1-mini",
            "o1-mini-2024-09-12",
            "o3-mini",
            "o3-mini-2025-01-31",
        ],
        online_models=[
            # TODO
            # "gpt-4o-realtime-preview",
            # "gpt-4o-realtime-preview-2024-10-01",
        ],
        required_config_fields=["api_key"],
    )


@RemoteLanguageModel.register("azure-openai")
class AzureOpenAIModel(OpenAICompatibleModel):
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions
    # noqa: https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2024-04-01-preview/inference.json
    _API_TEMPLATE = "{endpoint}/openai/deployments/{deployment_name}/chat/completions"

    # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/model-retirements#deprecated-models
    # "gpt-35-turbo-0613"/"gpt-35-turbo-0301" 已于 2025-02-13 过期,

    # NOTE: azure openai 的 deployment_name 可以和 model name 不一样，
    #       见 https://github.com/PrefectHQ/marvin/issues/842
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/model-retirements#current-models
    META = RemoteLanguageModelMetaInfo(
        language_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-instruct",
            "gpt-3.5-turbo-instruct-0914",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-32k-0314",
            "o1-mini",
            "o1-preview",
            "o3-mini",
        ],
        visual_language_models=[
            "gpt-4-vision-preview",
            "gpt-4-0215-preview",
            "gpt-4-1106-preview",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "o1",
        ],
        # https://platform.openai.com/docs/guides/function-calling
        tool_models=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-vision-preview",
            "gpt-4-0215-preview",
            "gpt-4-1106-preview",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "o1",
            "o1-preview",
            "o1-mini",
            "o3-mini",
        ],
        required_config_fields=[
            "api_key",
            "azure_endpoint",
            "azure_deployment_name",
        ],
    )
    REQUEST_BODY_CLS = AzureOpenAIRequestBody

    def _make_api_headers(self):
        return {"api-key": self.config.api_key.get_secret_value()}

    def _make_api_params(self):
        return {"api-version": self.config.azure_api_version}

    def _get_api_url(self):
        return self._API_TEMPLATE.format(
            endpoint=self.config.azure_endpoint,
            deployment_name=self.config.azure_deployment_name,
        )
