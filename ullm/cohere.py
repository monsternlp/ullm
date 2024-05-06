import json
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    Json,
    NonNegativeFloat,
    PositiveInt,
    confloat,
    conint,
    conlist,
    validate_call,
)

from .base import (
    AssistantMessage,
    ChatMessage,
    GenerateConfig,
    GenerationResult,
    HttpServiceModel,
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
    Tool,
    ToolCall,
    ToolChoice,
    ToolMessage,
    UserMessage,
)


class CohereChatMessage(BaseModel):
    role: Literal["SYSTEM", "USER", "CHATBOT"]
    message: str

    @classmethod
    def from_standard(cls, message: ChatMessage):
        if isinstance(ToolMessage, ToolMessage):
            raise NotImplementedError

        if isinstance(message, UserMessage):
            return cls(role="USER", message="\n".join([part.text for part in message.content]))

        if isinstance(message, AssistantMessage):
            return cls(role="CHATBOT", message=message.content or "")


class CohereWebSearchConnectorOptions(BaseModel):
    site: str


class CohereWebSearchConnector(BaseModel):
    id: Literal["web-search"] = "web-search"
    continue_on_failure: Optional[bool] = None
    options: Optional[CohereWebSearchConnectorOptions] = None


class CohereCustomConnector(BaseModel):
    id: str
    user_access_token: Optional[str] = None
    continue_on_failure: Optional[bool] = None
    options: Optional[Dict] = None


class CohereDocument(BaseModel, extra="allow"):
    id: Optional[str] = Field(default_factory=lambda: uuid4().hex)


class CohereToolParameterDefinition(BaseModel):
    type: str
    description: Optional[str] = None
    required: Optional[bool] = None


class CohereTool(BaseModel):
    name: str
    description: str
    parameter_definitions: Optional[Dict[str, CohereToolParameterDefinition]] = None

    @classmethod
    def from_standard(cls, tool: Tool):
        parameter_definitions = {}
        for argument in tool.function.arguments or []:
            parameter_definitions[argument.name] = argument.model_dump()

        return cls(
            name=tool.function.name,
            description=tool.function.description,
            parameter_definitions=parameter_definitions,
        )


class CohereToolCall(BaseModel):
    name: str
    parameters: Optional[Union[Json, Dict[str, Any]]] = {}

    def to_standard(self):
        return ToolCall(type="function", function={"name": self.name, "arguments": self.parameters})


class CohereToolResult(BaseModel):
    call: CohereToolCall
    outputs: conlist(Dict[str, Any], min_length=1)


class CohereRequestBody(BaseModel):
    # reference: https://docs.cohere.com/reference/chat
    message: str
    model: str
    stream: Optional[bool] = None
    preamble: Optional[str] = None
    chat_history: Optional[List[CohereChatMessage]] = None
    conversation_id: Optional[str] = None
    prompt_truncation: Optional[Literal["AUTO", "OFF"]] = None
    connectors: Optional[List[Union[CohereWebSearchConnector, CohereCustomConnector]]] = None
    search_queries_only: Optional[bool] = None
    documents: Optional[List[CohereDocument]] = None
    citation_quality: Optional[Literal["accurate", "fast"]] = None
    temperature: Optional[NonNegativeFloat] = None
    max_tokens: Optional[PositiveInt] = None
    max_input_tokens: Optional[PositiveInt] = None
    k: Optional[conint(ge=0, le=500)] = None
    p: Optional[confloat(ge=0.01, le=0.99)] = None
    seed: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    frequency_penalty: Optional[confloat(ge=0.0, le=1.0)] = None
    presence_penalty: Optional[confloat(ge=0.0, le=1.0)] = None
    tools: Optional[List[CohereTool]] = None
    tool_results: Optional[List[CohereToolResult]] = None


class CohereCitation(BaseModel):
    start: int
    end: int
    text: str
    document_ids: conlist(str, min_length=1)


class CohereSearchQuery(BaseModel):
    text: str
    generation_id: str


class CohereSearchResult(BaseModel):
    search_query: Optional[CohereSearchQuery] = None
    connector: Union[CohereWebSearchConnector, CohereCustomConnector]
    document_ids: Optional[List[str]] = None
    error_message: Optional[str] = Field(None, description="An error message if the search failed.")
    continue_on_failure: Optional[bool] = None


class CohereResponseAPIVersion(BaseModel):
    version: str
    is_deprecated: Optional[bool] = None
    is_experimental: Optional[bool] = None


class CohereResponseBilledUnits(BaseModel):
    input_tokens: int
    output_tokens: int
    search_units: Optional[int] = None
    classifications: Optional[int] = None


class CohereResponseTokens(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class CohereResponseMeta(BaseModel):
    api_version: Optional[CohereResponseAPIVersion] = None
    billed_units: Optional[CohereResponseBilledUnits] = None
    tokens: CohereResponseTokens
    warning: Optional[List[str]] = None


class CohereResponseBody(BaseModel):
    text: str
    chat_history: List[CohereChatMessage]
    generation_id: Optional[str] = None
    citations: Optional[List[CohereCitation]] = None
    documents: Optional[List[CohereDocument]] = None
    is_search_required: Optional[bool] = None
    search_queries: Optional[List[CohereSearchQuery]] = None
    search_results: Optional[List[CohereSearchResult]] = None
    finish_reason: Literal[
        "COMPLETE", "ERROR", "ERROR_TOXIC", "ERROR_LIMIT", "USER_CANCEL", "MAX_TOKENS"
    ]
    tool_calls: Optional[List[CohereToolCall]] = None
    meta: CohereResponseMeta

    def to_standard(self, model: str = None):
        tool_calls = (
            [tool_call.to_standard() for tool_call in self.tool_calls] if self.tool_calls else None
        )
        total_tokens = None
        if self.meta.tokens.input_tokens is not None and self.meta.tokens.output_tokens is not None:
            total_tokens = self.meta.tokens.input_tokens + self.meta.tokens.output_tokens

        return GenerationResult(
            model=model,
            stop_reason=self.finish_reason,
            content=self.text,
            tool_calls=tool_calls,
            input_tokens=self.meta.tokens.input_tokens,
            output_tokens=self.meta.tokens.output_tokens,
            total_tokens=total_tokens,
        )


@RemoteLanguageModel.register("cohere")
class CohereModel(HttpServiceModel):
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.cohere.ai/v1/chat",
        language_models=[
            "command",
            "command-nightly",
            "command-light",
            "command-light-nightly",
            "command-r",
            "command-r-plus",
            "command-online",
            "command-nightly-online",
            "command-light-online",
            "command-light-nightly-online",
            "command-r-online",
            "command-r-plus-online",
        ],
        visual_language_models=[],
        tool_models=[
            "command-r",
            "command-r-online",
            "command-r-plus",
            "command-r-plus-online",
        ],
        online_models=[
            "command-online",
            "command-nightly-online",
            "command-light-online",
            "command-light-nightly-online",
            "command-r-online",
            "command-r-plus-online",
        ],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = CohereRequestBody
    RESPONSE_BODY_CLS = CohereResponseBody

    def _make_api_headers(self):
        return {"Authorization": f"Bearer {self.config.api_key.get_secret_value()}"}

    @classmethod
    def _convert_message(cls, message: ChatMessage):
        pass

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        merged_messages = []
        tool_results = {}
        for message in messages:
            if isinstance(message, UserMessage):
                merged_messages.append(CohereChatMessage.from_standard(message))
            elif isinstance(message, AssistantMessage):
                if message.tool_calls:
                    if not merged_messages:
                        continue

                    index = len(merged_messages) - 1
                    tool_results[index] = {"calls": [], "results": {}}
                    for tool_call in message.tool_calls:
                        tool_results[index]["calls"].append(
                            {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "parameters": tool_call.function.arguments,
                            }
                        )
                else:
                    merged_messages.append(CohereChatMessage.from_standard(message))
            elif isinstance(message, ToolMessage):
                index = len(merged_messages) - 1
                if index < 0 or index not in tool_results:
                    continue

                output = None
                try:
                    output = json.loads(message.tool_result)
                except (TypeError, json.JSONDecodeError):
                    output = {"content": message.tool_result}

                tool_results[index]["results"].setdefault(message.tool_call_id, []).append(output)

        result = {"message": merged_messages[-1].message, "chat_history": merged_messages[:-1]}
        final_index = len(merged_messages) - 1
        if final_index not in tool_results:
            return result

        result["tool_results"] = []
        for tool_call in tool_results[final_index]["calls"]:
            call_outputs = tool_results[final_index]["results"].get(tool_call["id"], [])
            if not call_outputs:
                continue

            result["tool_results"].append(
                {
                    "call": {
                        "name": tool_call["name"],
                        "parameters": tool_call["parameters"],
                    },
                    "outputs": call_outputs,
                }
            )

        return result

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        if tools:
            tools = [CohereTool.from_standard(tool) for tool in tools]

        return {"tools": tools}

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        connectors = None
        if self.is_online_model():
            connectors = [CohereWebSearchConnector(id="web-search")]

        return {
            "model": self.model.replace("-online", ""),
            "max_tokens": config.max_output_tokens or self.config.max_output_tokens,
            "stop_sequences": config.stop_sequences or self.config.stop_sequences,
            "temperature": config.temperature or self.config.temperature,
            "p": config.top_p or self.config.top_p,
            "k": config.top_k or self.config.top_k,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "connectors": connectors,
        }
