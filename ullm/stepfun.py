from typing import Any, Optional

from pydantic import Field

from .base import (
    RemoteLanguageModel,
    RemoteLanguageModelMetaInfo,
)
from .openai import OpenAICompatibleModel, OpenAIRequestBody


class StepFunRequestBody(OpenAIRequestBody):
    # reference: https://platform.stepfun.com/docs/Chat/chat-completion-create
    ## excluded parameters
    logit_bias: Optional[Any] = Field(None, exclude=True)
    logprobs: Optional[Any] = Field(None, exclude=True)
    top_logprobs: Optional[Any] = Field(None, exclude=True)
    response_format: Optional[Any] = Field(None, exclude=True)
    seed: Optional[Any] = Field(None, exclude=True)
    tools: Optional[Any] = Field(None, exclude=True)
    tool_choice: Optional[Any] = Field(None, exclude=True)
    user: Optional[Any] = Field(None, exclude=True)


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
        ],
        visual_language_models=[
            "step-1v-8k",
            "step-1v-32k",
        ],
        tool_models=[],
        online_models=[],
        required_config_fields=["api_key"],
    )
    REQUEST_BODY_CLS = StepFunRequestBody
