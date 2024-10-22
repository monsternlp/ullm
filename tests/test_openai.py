import base64
import os

import pytest

from ullm import AssistantMessage, LanguageModel, ToolMessage, UserMessage
from ullm.openai import (
    OpenAIAssistantMessage,
    OpenAICompatibleModel,
    OpenAIModel,
    OpenAIToolMessage,
    OpenAIUserMessage,
)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DATA = None
IMAGE_BASE64_DATA = None
IMAGE_FILE = os.path.join(WORK_DIR, "SSU_Kirby_artwork.png")
with open(IMAGE_FILE, "rb") as f:
    IMAGE_DATA = f.read()
    IMAGE_BASE64_DATA = base64.b64encode(IMAGE_DATA).decode("utf-8")


@pytest.mark.parametrize(
    ("config", "cls"),
    [
        (
            {
                "type": "remote",
                "provider": "openai-compatible",
                "model": "mymodel",
                "api_url": "http://localhost:8080/api/v1/chat/completions",
                "is_visual_model": False,
                "is_online_model": False,
                "is_tool_model": False,
            },
            OpenAICompatibleModel,
        ),
        (
            {
                "type": "remote",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "sk-************************************************",
            },
            OpenAIModel,
        ),
    ],
)
def test_load(config, cls):
    model = LanguageModel.from_config(config)
    assert isinstance(model, cls)


@pytest.mark.parametrize(
    "config",
    [
        {
            "type": "remote",
            "provider": "openai-compatible",
            "model": "mymodel",
        },
        {
            "type": "remote",
            "provider": "openai",
            "model": "gpt-4",
        },
        {
            "type": "remote",
            "provider": "openai",
            "model": "not-a-openai-model",
            "api_key": "sk-************************************************",
        },
    ],
)
def test_load_error(config):
    with pytest.raises(ValueError):
        _ = LanguageModel.from_config(config)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (
            AssistantMessage(content="A assistant response"),
            OpenAIAssistantMessage(content="A assistant response"),
        ),
        (
            AssistantMessage(
                tool_calls=[
                    {
                        "id": "id",
                        "type": "function",
                        "function": {
                            "name": "func1",
                            "arguments": '{"arg1": "val1", "arg2": "val2"}',
                        },
                    }
                ]
            ),
            OpenAIAssistantMessage(
                tool_calls=[
                    {
                        "id": "id",
                        "type": "function",
                        "function": {
                            "name": "func1",
                            "arguments": '{"arg1": "val1", "arg2": "val2"}',
                        },
                    }
                ]
            ),
        ),
        (
            UserMessage(content=[{"type": "text", "text": "hello"}]),
            OpenAIUserMessage(content="hello"),
        ),
        (
            UserMessage(
                content=[
                    {
                        "type": "image",
                        "url": "https://upload.wikimedia.org/wikipedia/zh/2/2d/SSU_Kirby_artwork.png",
                    }
                ]
            ),
            OpenAIUserMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/zh/2/2d/SSU_Kirby_artwork.png"
                        },
                    }
                ]
            ),
        ),
        (
            UserMessage(
                content=[
                    {
                        "type": "image",
                        "path": IMAGE_FILE,
                    }
                ]
            ),
            OpenAIUserMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{IMAGE_BASE64_DATA}"},
                    }
                ]
            ),
        ),
        (
            UserMessage(
                content=[
                    {
                        "type": "image",
                        "path": IMAGE_FILE,
                    },
                    {"type": "text", "text": "Describe this image."},
                ]
            ),
            OpenAIUserMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{IMAGE_BASE64_DATA}"},
                    },
                    {"type": "text", "text": "Describe this image."},
                ]
            ),
        ),
        (
            ToolMessage(tool_call_id="id1", tool_name="func1", tool_result="success"),
            OpenAIToolMessage(tool_call_id="id1", content="success"),
        ),
    ],
)
def test_convert_message(source, target):
    converted = OpenAICompatibleModel._convert_message(source)
    assert type(converted) is type(target)
    assert converted and converted.model_dump() == target.model_dump()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature for a specific location",
            "arguments": [
                {
                    "name": "location",
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA",
                },
            ],
        },
    }
]
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA",
                    },
                },
                "required": ["location"],
            },
        },
    }
]


@pytest.mark.parametrize(
    ("model_config", "messages", "config", "system", "api_body"),
    [
        (
            {
                "type": "remote",
                "provider": "openai-compatible",
                "model": "mymodel",
                "max_output_tokens": 1024,
                "api_url": "http://localhost:8080/api/v1/chat/completions",
                "is_visual_model": False,
                "is_online_model": False,
                "is_tool_model": False,
            },
            [{"role": "user", "content": "hello"}],
            {"temperature": 0.1},
            "Act as a child.",
            {
                "messages": [
                    {"role": "system", "content": "Act as a child."},
                    {"role": "user", "content": "hello"},
                ],
                "model": "mymodel",
                "max_tokens": 1024,
                "n": 1,
                "response_format": {"type": "text"},
                "stream": False,
                "temperature": 0.1,
            },
        ),
        (
            {
                "type": "remote",
                "provider": "openai-compatible",
                "model": "mymodel",
                "max_output_tokens": 1024,
                "api_url": "http://localhost:8080/api/v1/chat/completions",
                "is_visual_model": False,
                "is_online_model": False,
                "is_tool_model": False,
            },
            [{"role": "user", "content": "hello"}],
            {"temperature": 0.1, "max_output_tokens": 4096},
            "Act as a child.",
            {
                "messages": [
                    {"role": "system", "content": "Act as a child."},
                    {"role": "user", "content": "hello"},
                ],
                "model": "mymodel",
                "max_tokens": 4096,
                "n": 1,
                "response_format": {"type": "text"},
                "stream": False,
                "temperature": 0.1,
            },
        ),
        (
            {
                "type": "remote",
                "provider": "openai-compatible",
                "model": "mymodel",
                "api_url": "http://localhost:8080/api/v1/chat/completions",
                "is_visual_model": False,
                "is_online_model": False,
                "is_tool_model": True,
            },
            [{"role": "user", "content": "hello"}],
            {
                "temperature": 0.1,
                "tools": TOOLS,
                "tool_choice": {
                    "mode": "any",
                    "functions": ["get_current_temperature"],
                },
            },
            None,
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "mymodel",
                "n": 1,
                "response_format": {"type": "text"},
                "stream": False,
                "temperature": 0.1,
                "tools": OPENAI_TOOLS,
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_current_temperature"},
                },
            },
        ),
    ],
)
def test_make_api_body(model_config, messages, config, system, api_body):
    model = LanguageModel.from_config(model_config)
    assert model._make_api_body(messages, config, system) == api_body
