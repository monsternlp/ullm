import pytest

from ullm import (
    AssistantMessage,
    GenerationResult,
    HttpServiceModel,
    OpenAIRequestBody,
    OpenAIResponseBody,
    RemoteLanguageModelConfig,
)
from ullm.types import ToolCall


class MockHttpServiceModel(HttpServiceModel):
    def __init__(self, config, mock_response: GenerationResult):
        super().__init__(config)
        self.mock_response = mock_response

    def _make_api_headers(self):
        return {}

    def _convert_message(self, message):
        pass

    def chat(self, messages, config=None, system=None):
        return self.mock_response


@pytest.fixture
def mock_model_config():
    return RemoteLanguageModelConfig(
        type="remote",
        provider="mock-provider",
        model="mock-model",
    )


def test_chat_as_openai(mock_model_config):
    request_body_dict = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "mock-model",
        "temperature": 0.5,
    }

    request_body = OpenAIRequestBody.model_validate(request_body_dict)

    mock_generation_result = GenerationResult(
        model="mock-model",
        stop_reason="stop",
        content="Hi there!",
        tool_calls=None,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )

    model = MockHttpServiceModel(config=mock_model_config, mock_response=mock_generation_result)

    response_body = model.chat_as_openai(**request_body.model_dump())

    assert isinstance(response_body, OpenAIResponseBody)
    assert response_body.model == "mock-model"
    assert len(response_body.choices) == 1
    choice = response_body.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.content == "Hi there!"
    assert choice.message.tool_calls is None
    assert response_body.usage.prompt_tokens == 10
    assert response_body.usage.completion_tokens == 5
    assert response_body.usage.total_tokens == 15


def test_chat_as_openai_with_tools(mock_model_config):
    request_body_dict = {
        "messages": [{"role": "user", "content": "What's the weather like in Boston?"}],
        "model": "mock-model",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    }

    request_body = OpenAIRequestBody.model_validate(request_body_dict)

    tool_calls = [
        ToolCall(
            id="call_123",
            type="function",
            function={
                "name": "get_current_weather",
                "arguments": {"location": "Boston, MA"},
            },
        )
    ]

    mock_assistant_message = AssistantMessage(tool_calls=tool_calls)

    mock_generation_result = GenerationResult(
        model="mock-model",
        stop_reason="tool_calls",
        content=None,
        tool_calls=mock_assistant_message.tool_calls,
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
    )

    model = MockHttpServiceModel(config=mock_model_config, mock_response=mock_generation_result)
    response_body = model.chat_as_openai(**request_body.model_dump())

    assert isinstance(response_body, OpenAIResponseBody)
    assert response_body.model == "mock-model"
    assert len(response_body.choices) == 1
    choice = response_body.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content is None
    assert len(choice.message.tool_calls) == 1
    tool_call = choice.message.tool_calls[0]
    assert tool_call.id == "call_123"
    assert tool_call.function.name == "get_current_weather"
    assert tool_call.function.arguments == '{"location": "Boston, MA"}'
    assert response_body.usage.prompt_tokens == 20
    assert response_body.usage.completion_tokens == 10
    assert response_body.usage.total_tokens == 30
