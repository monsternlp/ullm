from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conint, conlist, validate_call

from .base import (
    AssistantMessage,
    ChatMessage,
    GenerateConfig,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    TextPart,
    ToolMessage,
    UserMessage,
)
from .openai import (
    OpenAICompatibleModel,
    OpenAIRequestBody,
)


class BaichuanChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @classmethod
    def from_standard(cls, message: ChatMessage):
        if isinstance(message, AssistantMessage):
            content = message.content
            if message.tool_calls:  # NOTE: API 文档中未定义相关行为
                content = ""
                for tool_call in message.tool_calls:
                    content += (
                        f"You should call the function `{tool_call.function.name}` with arguments: "
                        f"{tool_call.function.arguments}\n"
                    )

            return BaichuanChatMessage(role="assistant", content=content.strip())

        if isinstance(message, UserMessage):
            content = ""
            for part in message.content:
                if isinstance(part, TextPart):
                    content += part.text + "\n"
                else:
                    pass

            return BaichuanChatMessage(role="user", content=content.strip())

        if isinstance(message, ToolMessage):
            content = (
                f"I called the function `{message.tool_name}` and "
                f"the response of that function is: {message.tool_result}"
            )
            return cls(role="user", content=content)


class BaichuanRetrievalObject(BaseModel):
    kb_ids: List[str]
    answer_mode: Optional[Literal["knowledge-base-only"]] = None


class BaichuanTool(BaseModel):
    type: Literal["retrieval"] = "retrieval"
    retrieval: BaichuanRetrievalObject


class BaichuanRequestBody(OpenAIRequestBody):
    # reference: https://platform.baichuan-ai.com/docs/api#12
    ## excluded parameters
    frequency_penalty: Optional[Any] = Field(None, exclude=True)
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(None, exclude=True)
    presence_penalty: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    stop: Optional[Any] = Field(None, exclude=True)
    tool_choice: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)

    ## different parameters
    messages: conlist(BaichuanChatMessage, min_length=1)
    tools: Optional[List[BaichuanTool]] = None

    ## Baichuan-specific parameters
    top_k: Optional[conint(ge=0, le=20)] = None
    with_search_enhance: Optional[bool] = False


@RemoteLanguageModel.register("baichuan")
class BaichuanModel(OpenAICompatibleModel):
    # reference: https://platform.baichuan-ai.com/docs/api
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.baichuan-ai.com/v1/chat/completions",
        language_models=[
            "Baichuan2-Turbo",
            "Baichuan2-Turbo-192k",
            "Baichuan2-Turbo-online",
            "Baichuan2-Turbo-192k-online",
            "Baichuan3-Turbo",
            "Baichuan3-Turbo-128k",
            "Baichuan3-Turbo-online",
            "Baichuan3-Turbo-128k-online",
            "Baichuan4",
            "Baichuan4-online",
        ],
        visual_language_models=[],
        tool_models=[
            "Baichuan3-Turbo",
            "Baichuan3-Turbo-128k",
            "Baichuan4",
        ],
        online_models=[
            "Baichuan2-Turbo-online",
            "Baichuan2-Turbo-192k-online",
            "Baichuan3-Turbo-online",
            "Baichuan3-Turbo-128k-online",
            "Baichuan4-online",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = BaichuanRequestBody

    @classmethod
    @validate_call
    def _convert_message(cls, message: ChatMessage) -> BaichuanChatMessage:
        return BaichuanChatMessage.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = [self._convert_message(message) for message in messages]
        if system:
            # NOTE: 百川不支持 system 参数
            messages = [BaichuanChatMessage(role="user", content=system)] + messages

        return {"messages": messages}

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "model": self.model.replace("-online", ""),
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "stop": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "top_p": config.top_p or self.config.top_p,
            "top_k": config.top_k or self.config.top_k,
            "with_search_enhance": self.is_online_model(),
        }
