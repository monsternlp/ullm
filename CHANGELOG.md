# CHANGELOG

## v0.9.0

Added:

- Supported extra config with `GenerateConfig.extra`

## v0.8.1

Fixed

- Fixed `HubConfig`: ignore extra fields in `.env`

## v0.8.0

Added

- Implemented `TencentLKEModel`


Changed

- Updated supported models

## v0.7.0

Added

- Implemented `LanguageModel.is_ready` to check model status

## v0.6.0

Added

- Implemented `ByteDanceModel`
- Implemented `TencentModel`


Changed

- Updated supported models
- Supported `redis` backend in `ModelHub`


## v0.5.1

Fixed

- Fixed `GenerateConfig`: remove all default values, this fix ensures that only parameters explicitly set by the user are passed to the API

## v0.5.0

Changed

- Supported 5 new models in `AlibabaModel`: `qwen2-57b-a14b-instruct`, `qwen2-72b-instruct`, `qwen2-7b-instruct`, `qwen2-0.5b-instruct`
- Supported new model `Qwen/Qwen2-72B-Instruct` in `TogetherAIModel`
- Updated supported models in `OpenRouterModel`:

  - Supported 6 new models: `dolphin-mixtral-8x22b`, `mistral-7b-instruct-v0.1`, `mistral-7b-instruct-v0.2`, `mistral-7b-instruct-v0.3`, `openchat-8b`, `qwen-2-72b-instruct`
  - Removed 2 models: `cinematika-7b`, `cinematika-7b:free`

## v0.4.0

Added

- Implemented `ModelHub` to manage models
- Implemented new command `register-model` to register model in `ModelHub`
- Implemented new command `list-models` to list models registered in `ModelHub`

Fixed

- Customized serializer for `SecretStr` fields in `RemoteLanguageModelConfig`

Changed

- Renamed command `list-models` to `list-supported-models`
- Optimized command `chat` to support loading model from `ModelHub`


## v0.3.1

Fixed

- Fixed typo in `README.md`
- Fixed `config` parameter of`IflyTekModel.chat`
- Fixed validator of `ZhipuAITool`
- Fixed `ZhipuAIModel._convert_tools`
- Fixed `OpenRouter`: add model prefix in `_convert_generation_config`
- Fixed validator of `ToolCall`

## v0.3.0

Changed

- Supported 10 new models in `GoogleModel`: `gemini-1.0-pro`, `gemini-1.0-pro-001`, `gemini-1.0-pro-latest`, `gemini-1.0-pro-vision`, `gemini-1.0-pro-vision-latest`, `gemini-1.5-flash`, `gemini-1.5-flash-001`, `gemini-1.5-flash-latest`, `gemini-1.5-pro`, `gemini-1.5-pro-001`
- Updated supported models in `OpenRouterModel`:

  - Supported 39 new models: `olmo-7b-instruct`, `claude-1`, `claude-1.2`, `claude-instant-1.0`, `claude-instant-1.1`, `deepseek-chat`, `deepseek-coder`, `gemini-flash-1.5`, `zephyr-7b-beta`, `psyfighter-13b`, `bagel-34b`, `llava-yi-34b`, `llama-3-8b`, `llama3-70b`, `llama-3-8b-instruct:free`, `llama-guard-2-8b`, `phi-3-medium-128k-instruct`, `phi-3-medium-128k-instruct:free`, `phi-3-mini-128k-instruct`, `phi-3-mini-128k-instruct:free`, `llama-3-lumimaid-70b`, `hermes-2-pro-llama-3-8b`, `gpt-3.5-turbo-0301`, `gpt-3.5-turbo-0613`, `gpt-3.5-turbo-1106`, `gpt-4-0314`, `gpt-4-1106-preview`, `gpt-4-32k-0314`, `gpt-4o-2024-05-13`, `llama-3-sonar-large-32k-chat`, `llama-3-sonar-large-32k-online`, `llama-3-sonar-small-32k-chat`, `llama-3-sonar-small-32k-online`, `qwen-110b-chat`, `qwen-14b-chat`, `qwen-32b-chat`, `qwen-4b-chat`, `qwen-72b-chat`, `qwen-7b-chat`
  - Removed 9 models: `wizardlm-2-8x22b:nitro`, `pplx-70b-chat`, `pplx-70b-online`, `pplx-7b-chat`, `pplx-7b-online`, `sonar-medium-chat`, `sonar-medium-online`, `sonar-small-chat`, `sonar-small-online`

- Updated supported models in `ZeroOneAIModel`

  - Supported 9 new models: `yi-large`, `yi-medium`, `yi-medium-200k`, `yi-spark`, `yi-large-rag`, `yi-large-turbo`, `yi-large-preview`, `yi-large-rag-preview`, `yi-vision`
  - Removed 3 models: `yi-34b-chat-0205`, `yi-34b-chat-200k`, `yi-vl-plus`

- Updated supported models in `StepFunModel`

  - Supported 4 new models: `step-1-8k`, `step-1-128k`, `step-1-256k`, `step-1v-8k`
  - Removed 1 model: `step-1-200k`

- Supported new model `abab6.5g-chat` in `MiniMaxModel`
- Supported 2 new models in `AlibabaModel`: `qwen-max-0428`, `qwen-max-0428-online`
- Supported 5 new models in `BaiduModel`: `ERNIE-4.0-8K-Preview-0518`, `ERNIE-3.5-128K`, `Qianfan_Chinese_Llama_2-70B`, `ERNIE-4.0-8K-Preview-0518-online`, `ERNIE-3.5-128K-online`
- Supported 6 new models in `BaichuanModel`: `Baichuan3-Turbo`, `Baichuan3-Turbo-128k`, `Baichuan3-Turbo-online`, `Baichuan3-Turbo-128k-online`, `Baichuan4`, `Baichuan4-online`
- Updated `BaichuanModel` to support tools calling

## v0.2.0

Fixed

- Fixed typo in `README.md`
- `tabulate` is missing in dependencies


Added

- Supported `gpt-4o` in `OpenAIModel` and `OpenRouterModel`
- Implemented `SkyWorkModel`
- Implemented `CloudflareModel`
- Implemented `TogetherAIModel`

## v0.1.3

Fixed

- `jsonschema` is missing in dependencies.

## v0.1.2

Misc

- Removed useless print statements in code

## v0.1.1

Fixed

- `deepmerge` is missing in dependencies.

## v0.1.0

First release.
