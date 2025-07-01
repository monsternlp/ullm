import base64
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    confloat,
    conlist,
    model_validator,
    validate_call,
)

from .base import (
    AssistantMessage,
    ChatMessage,
    FunctionCall,
    GenerateConfig,
    GenerationResult,
    HttpServiceModel,
    ImagePart,
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


class GoogleBlobPart(BaseModel):
    mime_type: Literal["image/png", "image/jpeg", "image/heic", "image/heif", "image/webp"] = Field(
        ..., alias="mimeType"
    )
    data: str


class GoogleFunctionCallPart(BaseModel):
    name: str
    args: Dict[str, Any]

    def to_standard(self):
        return FunctionCall(name=self.name, arguments=self.args)


class GoogleFunctionResponsePart(BaseModel):
    name: str
    response: Dict[str, Any]


class GoogleFileDataPart(BaseModel):
    mime_type: Optional[str] = Field(None, alias="mimeType")
    file_uri: str = Field(..., alias="fileUri")


class GoogleContentPart(BaseModel):
    text: Optional[str] = None
    inline_data: Optional[GoogleBlobPart] = Field(None, alias="inlineData")
    function_call: Optional[GoogleFunctionCallPart] = Field(None, alias="functionCall")
    function_response: Optional[GoogleFunctionResponsePart] = Field(None, alias="functionResponse")
    file_data: Optional[GoogleFileDataPart] = Field(None, alias="fileData")

    @model_validator(mode="before")
    @classmethod
    def check_fields(cls, data):
        keys_with_value = []
        for key, value in data.items():
            if value is not None:
                keys_with_value.append(key)

        assert len(keys_with_value) == 1
        return data


class GoogleContent(BaseModel):
    # https://ai.google.dev/api/rest/v1beta/Content
    # and: https://ai.google.dev/gemini-api/docs/function-calling
    role: Literal["user", "model", "function"]
    parts: List[GoogleContentPart]

    @model_validator(mode="before")
    @classmethod
    def validate_role(cls, data):
        if data["role"] == "function":
            assert all(part.get("functionResponse") for part in data.get("parts", []))

        return data

    @classmethod
    @validate_call
    def from_standard(cls, message: ChatMessage):
        role, parts = None, []
        if isinstance(message, UserMessage):
            role = "user"
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append({"text": part.text})
                elif isinstance(part, ImagePart):
                    if part.data:
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": part.mime_type,
                                    "data": base64.b64encode(part.data).decode("utf-8"),
                                }
                            }
                        )
                    else:
                        parts.append(
                            {
                                "fileData": {
                                    "mimeType": part.mime_type,
                                    "fileUri": part.url,
                                }
                            }
                        )
        elif isinstance(message, AssistantMessage):
            role = "model"
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    parts.append(
                        {
                            "functionCall": {
                                "name": tool_call.function.name,
                                "args": tool_call.function.arguments,
                            }
                        }
                    )
            else:
                parts.append({"text": message.content})
        elif isinstance(message, ToolMessage):
            role = "function"
            response = message.tool_result
            try:
                response = json.loads(response)
            except (TypeError, json.JSONDecodeError):
                response = {"content": response}

            parts.append(
                {
                    "functionResponse": {
                        "name": message.tool_name,
                        "response": response,
                    }
                }
            )

        return cls(role=role, parts=parts)


class GoogleFunctionDeclaration(BaseModel):
    # https://ai.google.dev/api/rest/v1beta/Tool#FunctionDeclaration
    name: str
    description: str
    parameters: Optional[JsonSchemaObject] = None


class GoogleRetrievalConfig(BaseModel):
    mode: Literal["MODE_UNSPECIFIED", "MODE_DYNAMIC"]
    threshold: Optional[float] = Field(None, serialization_alias="dynamicThreshold")


class GoogleSearchRetrieval(BaseModel):
    retrieval_config: GoogleRetrievalConfig = Field(
        ..., serialization_alias="dynamicRetrievalConfig"
    )


class GoogleTool(BaseModel):
    # https://ai.google.dev/api/rest/v1beta/Tool
    function_declarations: Optional[List[GoogleFunctionDeclaration]] = Field(
        None, serialization_alias="functionDeclarations"
    )
    google_search_retrieval: Optional[GoogleSearchRetrieval] = Field(
        None, serialization_alias="googleSearchRetrieval"
    )
    code_execution: Optional[Any] = Field(None, serialization_alias="codeExecution")

    @classmethod
    def from_standard(cls, tools: List[Tool]) -> "GoogleTool":
        functions = []
        for tool in tools:
            parameters, required = {}, []
            for argument in tool.function.arguments or []:
                if argument.required:
                    required.append(argument.name)

                parameters[argument.name] = {
                    "type": argument.type,
                    "description": argument.description,
                }

            functions.append(
                GoogleFunctionDeclaration(
                    name=tool.function.name,
                    description=tool.function.description,
                    parameters={"type": "object", "properties": parameters, "required": required},
                )
            )

        return cls(function_declarations=functions)


class GoogleFunctionCallingConfig(BaseModel):
    mode: Optional[Literal["NONE", "AUTO", "ANY"]] = "AUTO"
    allowed_function_names: Optional[List[str]] = Field(
        None, serialization_alias="allowedFunctionNames"
    )


class GoogleToolConfig(BaseModel):
    # https://ai.google.dev/api/rest/v1beta/ToolConfig
    function_calling_config: GoogleFunctionCallingConfig = Field(
        ..., serialization_alias="functionCallingConfig"
    )

    @classmethod
    def from_standard(cls, tool_config: ToolChoice) -> "GoogleToolConfig":
        return cls(
            function_calling_config=GoogleFunctionCallingConfig(
                mode=tool_config.mode.upper(), allowed_function_names=tool_config.functions
            )
        )


class GoogleSafetySetting(BaseModel):
    # https://ai.google.dev/api/rest/v1beta/SafetySetting
    category: Literal[
        "HARM_CATEGORY_DEROGATORY",
        "HARM_CATEGORY_TOXICITY",
        "HARM_CATEGORY_VIOLENCE",
        "HARM_CATEGORY_SEXUAL",
        "HARM_CATEGORY_MEDICAL",
        "HARM_CATEGORY_DANGEROUS",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    threshold: Literal[
        "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "BLOCK_NONE"
    ]


class GoogleGenerationConfig(BaseModel):
    # https://ai.google.dev/api/rest/v1beta/GenerationConfig
    stop_sequences: Optional[List[str]] = Field(None, serialization_alias="stopSequences")
    response_mime_type: Literal["text/plain", "application/json"] = Field(
        "text/plain", serialization_alias="responseMimeType"
    )
    candidate_count: Optional[PositiveInt] = Field(1, serialization_alias="candidateCount")
    max_output_tokens: Optional[PositiveInt] = Field(None, serialization_alias="maxOutputTokens")
    temperature: Optional[confloat(ge=0.0, le=2.0)] = None
    top_p: Optional[confloat(ge=0.0, le=1.0)] = Field(None, serialization_alias="topP")
    top_k: Optional[PositiveInt] = Field(None, serialization_alias="topK")


class GoogleRequestBody(BaseModel):
    # https://platform.openai.com/docs/api-reference/chat/create
    contents: List[GoogleContent]
    tools: Optional[List[GoogleTool]] = None
    tool_config: Optional[GoogleToolConfig] = Field(None, serialization_alias="toolConfig")
    safety_settings: Optional[List[GoogleSafetySetting]] = Field(
        None, serialization_alias="safetySettings"
    )
    system_instruction: Optional[GoogleContent] = Field(
        None, serialization_alias="systemInstruction"
    )
    generation_config: Optional[GoogleGenerationConfig] = Field(
        None, serialization_alias="generationConfig"
    )


class GoogleSafetyRating(BaseModel):
    category: Literal[
        "HARM_CATEGORY_DEROGATORY",
        "HARM_CATEGORY_TOXICITY",
        "HARM_CATEGORY_VIOLENCE",
        "HARM_CATEGORY_SEXUAL",
        "HARM_CATEGORY_MEDICAL",
        "HARM_CATEGORY_DANGEROUS",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    probability: Literal["NEGLIGIBLE", "LOW", "MEDIUM", "HIGH"]
    blocked: Optional[bool] = None


class GoogleCandidate(BaseModel):
    # https://ai.google.dev/api/generate-content#v1beta.Candidate
    index: Optional[int] = None
    avg_logprobs: Optional[float] = Field(None, alias="avgLogprobs")
    logprobs_result: Optional[dict] = Field(None, alias="logprobsResult")
    grouding_metadata: Optional[dict] = Field(None, alias="groudingMetadata")  # TODO
    grouding_attributions: Optional[List[dict]] = Field(None, alias="groudingAttributions")  # TODO
    citation_metadata: Optional[dict] = Field(None, alias="citationMetadata")  # TODO
    content: GoogleContent
    finish_reason: Literal["STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"] = Field(
        ..., alias="finishReason"
    )
    safety_ratings: Optional[List[GoogleSafetyRating]] = Field(
        None,
        alias="safetyRatings",
        description="List of ratings for the safety of a response candidate.",
    )
    token_count: Optional[int] = Field(
        None, alias="tokenCount"
    )  # Google 目前没有按文档所说返回 tokenCount


class GooglePromptFeedback(BaseModel):
    block_reason: Literal["SAFETY", "OTHER"] = Field(..., alias="blockReason")
    safety_ratings: List[GoogleSafetyRating] = Field(
        ..., description="Ratings for safety of the prompt.", alias="safetyRatings"
    )


class GoogleUsageMetadata(BaseModel):
    prompt_token_count: int = Field(..., alias="promptTokenCount")
    cached_token_count: Optional[int] = Field(None, alias="cachedContentTokenCount")
    candidates_token_count: int = Field(..., alias="candidatesTokenCount")
    total_token_count: int = Field(..., alias="totalTokenCount")


class GoogleGenerateContentResponseBody(BaseModel):
    # https://ai.google.dev/api/rest/v1beta/GenerateContentResponse
    candidates: conlist(GoogleCandidate, min_length=1)
    prompt_feedback: Optional[GooglePromptFeedback] = Field(None, alias="promptFeedback")
    usage_metadata: GoogleUsageMetadata = Field(..., alias="usageMetadata")

    @validate_call
    def to_standard(self, model: str) -> GenerationResult:
        candidate = self.candidates[0]
        tool_calls = None
        if candidate.content.parts[0].function_call:
            tool_calls = [
                ToolCall(
                    type="function", function=candidate.content.parts[0].function_call.to_standard()
                )
            ]

        return GenerationResult(
            model=model,
            stop_reason=candidate.finish_reason,
            content=candidate.content.parts[0].text,
            input_tokens=self.usage_metadata.prompt_token_count,
            output_tokens=self.usage_metadata.candidates_token_count,
            total_tokens=self.usage_metadata.total_token_count,
            tool_calls=tool_calls,
        )


GOOGLE_API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


@RemoteLanguageModel.register("google")
class GoogleModel(HttpServiceModel):
    # https://ai.google.dev/gemini-api/docs/models/gemini
    _LANGUAGE_MODELS = []
    _VISUAL_LANGUAGE_MODELS = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash-8b-001",
        "gemini-1.5-flash-8b-latest",
        "gemini-1.5-pro",
        "gemini-1.5-pro-002",
        "gemini-1.5-pro-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite-preview-06-17",
    ]

    META = RemoteLanguageModelMetaInfo(
        model_api_url_mappings={
            model: GOOGLE_API_URL_TEMPLATE.format(model=model)
            for model in _LANGUAGE_MODELS + _VISUAL_LANGUAGE_MODELS
        },
        language_models=_LANGUAGE_MODELS,
        visual_language_models=_VISUAL_LANGUAGE_MODELS,
        tool_models=_VISUAL_LANGUAGE_MODELS,
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = GoogleRequestBody
    RESPONSE_BODY_CLS = GoogleGenerateContentResponseBody

    def _make_api_headers(self):
        return None

    def _make_api_params(self):
        return {"key": self.config.api_key.get_secret_value()}

    @classmethod
    @validate_call
    def _convert_message(cls, message: ChatMessage) -> GoogleContent:
        return GoogleContent.from_standard(message)

    @validate_call
    def _convert_messages(
        self,
        messages: conlist(ChatMessage, min_length=1),
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        contents = [self._convert_message(msg) for msg in messages]
        return {"contents": contents}

    @validate_call
    def _convert_tools(
        self, tools: Optional[List[Tool]] = None, tool_choice: Optional[ToolChoice] = None
    ) -> Dict[str, Any]:
        tools = None if not tools else [GoogleTool.from_standard(tools)]
        tool_config = None if not tool_choice else GoogleToolConfig.from_standard(tool_choice)
        return {"tools": tools, "tool_config": tool_config}

    def _convert_generation_config(
        self, config: GenerateConfig, system: Optional[str] = None
    ) -> Dict[str, Any]:
        response_mime_type = "text/plain"
        if config.response_format == "json_object":
            response_mime_type = "application/json"

        return {
            "system_instruction": system,
            "generation_config": GoogleGenerationConfig(
                stop_sequences=config.stop_sequences or self.config.stop_sequences,
                response_mime_type=response_mime_type,
                max_output_tokens=config.max_output_tokens or self.config.max_output_tokens,
                temperature=config.temperature or self.config.temperature,
                top_p=config.top_p or self.config.top_p,
                top_k=config.top_k or self.config.top_k,
            ),
        }
