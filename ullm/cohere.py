from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    confloat,
    conint,
)

from .base import (
    Citation,
    GenerateConfig,
    GenerationResult,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    TextPart,
)
from .openai import (
    OpenAICompatibleModel,
    OpenAIRequestBody,
    OpenAIToolCall,
)


class CohereDocument(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: uuid4().hex)
    data: Dict[str, str]


class CohereRequestBody(OpenAIRequestBody):
    # reference: https://docs.cohere.com/v2/reference/chat
    ## excluded parameters
    stop: Optional[Any] = Field(None, exclude=True)
    top_p: Optional[Any] = Field(None, exclude=True)
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    n: Optional[Any] = Field(None, exclude=True)
    tool_choice: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)

    ## different parameters
    frequency_penalty: Optional[confloat(ge=0.0, le=1.0)] = None
    presence_penalty: Optional[confloat(ge=0.0, le=1.0)] = None

    ## cohere specific parameters
    k: Optional[conint(ge=0, le=500)] = None
    p: Optional[confloat(ge=0.01, le=0.99)] = None
    documents: Optional[List[CohereDocument]] = None
    citation_options: Optional[Dict[Literal["mode"], Literal["ACCURATE", "FAST", "OFF"]]] = None
    safety_mode: Optional[Literal["CONTEXTUAL", "STRICT", "OFF"]] = None
    stop_sequences: Optional[List[str]] = None


class CohereResponseBilledUnits(BaseModel):
    input_tokens: int
    output_tokens: int
    search_units: Optional[int] = None
    classifications: Optional[int] = None


class CohereResponseTokens(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class CohereResponseUsage(BaseModel):
    billed_units: Optional[CohereResponseBilledUnits] = None
    tokens: CohereResponseTokens


class CohereResponseMessage(BaseModel):
    role: Literal["assistant"]
    tool_calls: Optional[List[OpenAIToolCall]] = None
    tool_plan: Optional[str] = None
    content: Optional[List[TextPart]] = None
    citations: Optional[List[Citation]] = None


class CohereResponseBody(BaseModel):
    id: str
    finish_reason: Literal["COMPLETE", "STOP_SEQUENCES", "MAX_TOKENS", "TOOL_CALL", "ERROR"]
    message: CohereResponseMessage
    usage: CohereResponseUsage

    def to_standard(self, model: str = None):
        tool_calls = (
            [tool_call.to_standard() for tool_call in self.message.tool_calls]
            if self.message.tool_calls
            else None
        )
        total_tokens = None
        if (
            self.usage.tokens.input_tokens is not None
            and self.usage.tokens.output_tokens is not None
        ):
            total_tokens = self.usage.tokens.input_tokens + self.usage.tokens.output_tokens

        content = "" if not self.message.content else self.message.content[0].text
        return GenerationResult(
            model=model,
            stop_reason=self.finish_reason,
            content=content,
            tool_calls=tool_calls,
            citations=self.message.citations,
            input_tokens=self.usage.tokens.input_tokens,
            output_tokens=self.usage.tokens.output_tokens,
            total_tokens=total_tokens,
        )


@RemoteLanguageModel.register("cohere")
class CohereModel(OpenAICompatibleModel):
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.cohere.ai/v2/chat",
        language_models=[
            "c4ai-aya-expanse-8b",
            "c4ai-aya-expanse-23b",
            "command",
            "command-nightly",
            "command-light",
            "command-light-nightly",
            "command-r",
            "command-r-03-2024",
            "command-r-08-2024",
            "command-r-plus",
            "command-r-plus-04-2024",
            "command-r-plus-08-2024",
            "command-r7b-12-2024",
        ],
        visual_language_models=[],
        tool_models=[
            "command-r",
            "command-r-03-2024",
            "command-r-08-2024",
            "command-r-plus",
            "command-r-plus-04-2024",
            "command-r-plus-08-2024",
            "command-r7b-12-2024",
        ],
        online_models=[],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = CohereRequestBody
    RESPONSE_BODY_CLS = CohereResponseBody

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "stop_sequences": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "p": config.top_p or self.config.top_p,
            "k": config.top_k or self.config.top_k,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
        }
