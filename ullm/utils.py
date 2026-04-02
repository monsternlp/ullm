import copy

from google import genai
from google.genai._transformers import process_schema
from pydantic import BaseModel


def to_gemini_json_schema(json_schema: dict | BaseModel | type[BaseModel]):
    """
    将 pydantic 模型或 JSON Schema 转换为 Gemini response_schema 支持的 JSON Schema 格式
    ref: https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#json_schema_support
    """

    if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
        json_schema = json_schema.model_json_schema()
    elif isinstance(json_schema, BaseModel):
        json_schema = type(json_schema).model_json_schema()
    else:
        json_schema = copy.deepcopy(json_schema)

    google_fake_client = genai.Client(api_key="fake")._api_client
    process_schema(json_schema, google_fake_client, order_properties=True)

    return json_schema
