<h1 align="center">ullm</h1>
<p align="center">A unified interface for local Large Language Models(LLM) and online LLM providers.</p>
<h4 align="center">
    <a href="https://pypi.org/project/ullm/" target="_blank">
        <img src="https://shields.io/pypi/v/ullm.svg" alt="PyPI Version">
    </a>
    <a href="https://github.com/monsternlp/ullm/actions/workflows/pre-commit.yaml" target="_blank">
        <img src="https://shields.io/github/actions/workflow/status/monsternlp/ullm/pre-commit.yaml?label=pre-commit" alt="Pre-commit status">
    </a>
    <a href="https://github.com/monsternlp/ullm/actions/workflows/publish.yaml" target="_blank">
        <img src="https://shields.io/github/actions/workflow/status/monsternlp/ullm/publish.yaml" alt="Build status">
    </a>
</h4>

ullm 希望能为本地模型以及众多在线 LLM 服务提供统一的调用方式，使得开发者能够无痛地在不同模型或 LLM 服务之间切换，而无需更改代码。

> [!NOTE]
> 本项目只专注于为不同 LLM 模型或服务的基础生成功能提供统一接口，包括通用的生成、聊天接口以及最基础的工具调用、视觉理解功能，在此之外的其他相关功能如 Finetuning、Prompt Engineering、Emebedding、RAG、Agent、TTS、ASR 本项目目前并不支持，将来也不会支持。

> [!WARNING]
> 本项目会尽可能地遵循语义版本，但在 1.0 版本前可能会发生不兼容的接口变动。

## 目录

<!-- TOC -->

- [功能与特性](#功能与特性)
- [支持模型](#支持模型)
  - [本地模型](#本地模型)
  - [在线服务](#在线服务)
- [安装](#安装)
- [使用](#使用)
  - [创建模型配置](#创建模型配置)
  - [实例化模型](#实例化模型)
  - [管理模型](#管理模型)
  - [设置生成参数](#设置生成参数)
  - [生成文本](#生成文本)
  - [聊天](#聊天)
- [命令行](#命令行)
  - [list-providers](#list-providers)
  - [list-supported-models](#list-supported-models)
  - [print-example](#print-example)
  - [chat](#chat)
  - [register-model](#register-model)
  - [list-models](#list-models)

<!-- /TOC -->

## 功能与特性

- 支持 OpenAI 等 24 个在线 LLM 服务，详见「[在线服务](#在线服务)」一节
- 支持和 OpenAI 接口兼容的自建服务
- 支持 Ollama API
- 配置化的使用方式，为所有不同模型及服务提供统一的初始化方式，详见「[使用](#使用)」一节
- [ ] 本地模型支持
- [ ] 归一化模型名称
- [ ] 为所有模型支持工具调用（对原本不支持的通过自定义 prompt 模板实现）
- [ ] 为不同模型适配对应的 tokenizer，以获得 tokens 数量解决某些 remote model 不返回 tokens 数量的问题
- [ ] 模型配置的管理
- [ ] 多模型路由
- [ ] 单元测试
- [ ] 完善文档
- [ ] 实现流式接口

## 支持模型

### 本地模型

TBD

### 在线服务


| 平台 | Provider ID            | 模型数量 | 视觉模型数量 | 支持工具调用的模型数量 | 联网模型数量 |
|------|-------------------|----------|--------------|------------------------|--------------|
| [零一万物](https://platform.lingyiwanwu.com/docs)     | 01ai              |        10 |            3 |                      1 |            0 |
| [阿里巴巴](https://help.aliyun.com/zh/dashscope/developer-reference/model-square/?spm=a2c4g.11186623.0.0.1cca23edHYSGqT)      | alibaba           |       131 |            29 |                      22 |            22 |
| [Anthropic](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)     | anthropic         |        8 |            8 |                      8 |            0 |
| [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions)     | azure-openai      |       27 |            11 |                     23 |            0 |
| [百川智能](https://platform.baichuan-ai.com/docs/api)     | baichuan          |        12 |            0 |                      5 |            6 |
| [百度](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu)     | baidu             |       39 |            0 |                     22 |           10 |
| [字节跳动豆包](https://www.volcengine.com/docs/82379/1298454)     | bytedance             |      13  |            3 |                     10 |           0 |
| [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/get-started/rest-api/) | cloudflare        |       36 |               0 |             1 |               0 |
| [Cohere](https://docs.cohere.com/reference/about)     | cohere            |       13 |            0 |                      7 |            0 |
| [DeepSeek](https://platform.deepseek.com/docs)     | deepseek          |        2 |            0 |                      1 |            0 |
| [Google](https://ai.google.dev/gemini-api/docs)     | google            |        18 |            18 |                      18 |            0 |
| [Groq](https://console.groq.com/docs/quickstart)     | groq              |        17 |            2 |                      9 |            0 |
| [科大讯飞](https://www.xfyun.cn/doc/spark/%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E.html)     | iflytek           |        12 |            0 |                      5 |            6 |
| [MiniMax](https://www.minimaxi.com/document/algorithm-concept?id=6433f37594878d408fc8295d)     | minimax           |        6 |            0 |                      6 |            3 |
| [Moonshot](https://platform.moonshot.cn/docs/api/chat#%E5%9F%BA%E6%9C%AC%E4%BF%A1%E6%81%AF)     | moonshot          |        3 |            0 |                      3 |            0 |
| [OpenAI](https://platform.openai.com/docs/api-reference/chat)      | openai            |       33 |            14 |                     29 |            0 |
| [OpenRouter](https://openrouter.ai/docs)     | openrouter        |      238 |           50 |                    238 |            3 |
| [Perplexity](https://docs.perplexity.ai/docs/getting-started)     | perplexity        |        5 |            0 |                      0 |            4 |
| [天工](https://model-platform.tiangong.cn/api-reference) | skywork           |        1 |               0 |             0 |               0 |
| [阶跃星辰](https://platform.stepfun.com/docs/Chat/chat-completion-create)     | stepfun           |        23 |            7 |                      16 |            8 |
| [腾讯混元](https://cloud.tencent.com/document/product/1729/105701) | tencent          |       31 |               4 |            4 |               14 |
| [腾讯大模型知识引擎](https://lke.cloud.tencent.com/lke/#/trialProduct) | tencent-lke       |        2 |               0 |             0 |               0 |
| [Together AI](https://docs.together.ai/docs/quickstart) | together          |       41 |               4 |            8 |               0 |
| [智谱](https://open.bigmodel.cn/dev/api)     | zhipu             |        15 |            2 |                      12 |            6 |
| OpenAI 接口兼容的服务     | openai-compatible |          |              |                        |              |
| Ollama API     | ollama            |          |              |                        |              |


## 安装

```shell
pip install ullm
```

## 使用

### 创建模型配置

示例:

```python
model_config = {
    # required fields
    "type": 'remote',
    "model": 'gpt-3.5-turbo',
    "provider": 'openai',
    "api_key": 'sk-************************************************',

    # optional fields
    "max_tokens": 4096,
    "max_input_tokens": 1024,
    "max_output_tokens": 1024,
    "temperature": 0.8,
    "top_p": 1.0,
    "top_k": 50,
    "stop_sequences": ['stop1', 'stop2'],
    "http_proxy": 'https://example-proxy.com',
}

```

模型配置中必须指定这三个字段

- type: 指定模型为本地模型(`local`)还是在线模型(`remote`)，目前仅实现了 remote
- provider: 指定模型提供方，见前面「[支持模型](#支持模型)」一节，或者你也可以通过命令行工具来或获取目前支持的在线服务提供方

  ```shell
  ullm list-providers
  ```

  会得到如下结果，使用其中的 name 作为配置文件中 `provider` 的值

  ```
  | name              |   models |   visual_models |   tool_models |   online_models |
  |-------------------|----------|-----------------|---------------|-----------------|
  | 01ai              |        3 |               1 |             0 |               0 |
  | alibaba           |       44 |               4 |             6 |               6 |
  | anthropic         |        3 |               3 |             3 |               0 |
  | azure-openai      |       20 |               4 |            11 |               0 |
  | baichuan          |        4 |               0 |             0 |               2 |
  | baidu             |       46 |               0 |            24 |              12 |
  | cohere            |       12 |               0 |             4 |               6 |
  | deepseek          |        2 |               0 |             0 |               0 |
  | google            |        3 |               2 |             2 |               0 |
  | groq              |        4 |               0 |             4 |               0 |
  | iflytek           |        4 |               0 |             2 |               0 |
  | minimax           |        5 |               0 |             5 |               0 |
  | moonshot          |        3 |               0 |             3 |               0 |
  | ollama            |          |                 |               |                 |
  | openai            |       20 |               4 |            11 |               0 |
  | openai-compatible |          |                 |               |                 |
  | openrouter        |      127 |              13 |           127 |               4 |
  | perplexity        |        7 |               0 |             0 |               2 |
  | stepfun           |        3 |               1 |             0 |               0 |
  | zhipu             |        5 |               1 |             4 |               2 |
  ```

- `model`: 指定要使用的模型名字

除这三个字段外，各不同平台的模型可能会有各自的一些必需字段（如 `api_key`），这些必需字段都在每个模型类的 META 中定义，如 `OpenAIModel`

```python
class OpenAIModel(OpenAICompatibleModel):
    META = RemoteLanguageModelMetaInfo(
        api_url="https://api.openai.com/v1/chat/completions",
        language_models=[
            # ......
        ],
        visual_language_models=[
            # ......
        ],
        # https://platform.openai.com/docs/guides/function-calling
        tool_models=[
            # ......
        ],
        required_config_fields=["api_key"],
    )
```

由于支持的平台众多，为方便起见，ullm 在命令行接口中提供了工具来生成示例代码供您修改

```shell
ullm print-example --provider openai
```

会得到如下输出

```
from ullm import LanguageModel

config = {
    # required fields
    "type": 'remote',
    "model": 'gpt-3.5-turbo',
    "provider": 'openai',
    "api_key": 'sk-************************************************',

    # optional fields
    "max_tokens": 4096,
    "max_input_tokens": 1024,
    "max_output_tokens": 1024,
    "temperature": 0.8,
    "top_p": 1.0,
    "Top_k": 50,
    "stop_sequences": ['stop1', 'stop2'],
    "http_proxy": 'https://example-proxy.com',
}
model = LanguageModel.from_config(config)
messages = [{"role": "user", "content": "Hello!"}]
res = model.chat(messages)
messages.append(res.to_message())
messages.append({"role": "user", "content": "Tell me a joke please!"})
res = model.chat(messages)
```

生成的示例代码中 `required_fields` 下方的配置项就是必须配置的。

### 实例化模型

所有模型的实例化统一通过 `LanguageModel` 这个类来进行，您不必关心不同模型的具体类名。

```python
from ullm import LanguageModel

model = LanguageModel.from_config(model_config)
```

如果配置不符合要求，这个过程中可能会报错，具体来说

- 如果配置中缺失了一些必需字段，会报错
- 如果模型名称不在支持的模型列表里，会报错

### 管理模型

`ullm` 实现了 `ModelHub` 来提供简易的模型管理，使用它可以

- 将模型实例的配置注册到 `ModelHub` 中

  ```python
  from ullm import LanguageModel, ModelHub

  config = {
      # required fields
      "type": 'remote',
      "model": 'gpt-3.5-turbo',
      "provider": 'openai',
      "api_key": 'sk-************************************************',

      # optional fields
      "max_tokens": 4096,
      "max_input_tokens": 1024,
      "max_output_tokens": 1024,
      "temperature": 0.8,
      "top_p": 1.0,
      "Top_k": 50,
      "stop_sequences": ['stop1', 'stop2'],
      "http_proxy": 'https://example-proxy.com',
  }
  model = LanguageModel.from_config(config)

  hub = ModelHub()
  hub.register_model(model, "openai:gpt-3.5-turbo")
  ```

  或者也可以使用命令行工具来注册模型

  ```shell
  ullm register-model --model-id "openai:gpt-3.5-turbo" --model-config-file openai.json
  ```

- 通过注册时分配的唯一性 Model ID 从 `ModelHub` 中获取一个模型实例来进行聊天

  ```python
  model = hub.get_model("openai:gpt-3.5-turbo")
  model.chat([{"role": "user", "content": "Hello"}])
  ```

默认情况下 `ModelHub` 会生成一个 `SQLite3` 的数据库文件 `$HOME/.ullm.db`，并在这个数据库中存储已注册的模型实例配置，若希望更改数据库文件路径或使用其他存储后端（如 `MySQL` 或 `PostPostgres` 或），可以通过不同方法来自定义存储类型和 URL：

- 实例化时直接指定存储类型和 URL

  - 使用 `SQLite3` 并更改数据库文件路径为 `/home/user/mymodels.db`

    ```python
    hub = ModelHub(hub_backend="rds", hub_db_url="sqlite:////home/user/my.db")
    ```

  - 使用 `MySQL`

    ```python
    hub = ModelHub(hub_backend="rds", hub_db_url="mysql://user:passwd@ip:port/my_db")
    ```

  - 使用 `Postgres`

    ```python
    hub = ModelHub(hub_backend="rds", hub_db_url="postgresql://postgres:my_password@localhost:5432/my_db")
    ```

  - 使用 `Redis`

    ```python
    hub = ModelHub(hub_backend="redis", hub_db_url="redis://localhost:6379/0", hub_redis_prefix="/ullm/model_hub/")
    ```

- 设置环境变量


  - 使用 `SQLite3` 并更改数据库文件路径为 `/home/user/mymodels.db`

    ```shell
    export ULLM_HUB_BACKEND=rds
    export ULLM_HUB_DB_URL=sqlite:////home/user/my.db
    ```

  - 使用 `MySQL`

    ```shell
    export ULLM_HUB_BACKEND=rds
    export ULLM_HUB_DB_URL=mysql://user:passwd@ip:port/my_db
    ```

  - 使用 `Postgres`

    ```shell
    export ULLM_HUB_BACKEND=rds
    export ULLM_HUB_DB_URL=postgresql://postgres:my_password@localhost:5432/my_db
    ```

  - 使用 `Redis`

    ```shell
    export ULLM_HUB_BACKEND=redis
    export ULLM_HUB_DB_URL=redis://localhost:6379/0
    export ULLM_HUB_REDIS_PREFIX=/ullm/model_hub/
    ```

- 设置 `.env` 环境变量文件

  - 使用 `SQLite3` 并更改数据库文件路径为 `/home/user/mymodels.db`

    ```
    ULLM_HUB_BACKEND=rds
    ULLM_HUB_DB_URL=sqlite:////home/user/my.db
    ```

  - 使用 `MySQL`

    ```
    ULLM_HUB_BACKEND=rds
    ULLM_HUB_DB_URL=mysql://user:passwd@ip:port/my_db
    ```

  - 使用 `Postgres`

    ```
    ULLM_HUB_BACKEND=rds
    ULLM_HUB_DB_URL=postgresql://postgres:my_password@localhost:5432/my_db
    ```

  - 使用 `Redis`

    ```
    ULLM_HUB_BACKEND=redis
    ULLM_HUB_DB_URL=redis://localhost:6379/0
    ULLM_HUB_REDIS_PREFIX=/ullm/model_hub/
    ```

### 设置生成参数

示例：

```python
generate_config = {
    "temperature": 0.7,
    "max_tokens": None,
    "max_input_tokens": None,
    "max_output_tokens": 1024,
    "top_p": None,
    "top_k": None,
    "stop_sequences": ["stop1", "stop2"],
    "frequency_penalty": None,
    "presence_penalty": None,
    "repetition_penalty": None,
    "tools": None,
    "tool_choice": None,
}
```


ullm 支持在生成时指定部分生成参数以覆盖模型初始化时指定的一些生成参数，目前的生成参数支持这些

- `temperature`: 指定模型使用温度采样生成方法，并控制生成随机程度
- `max_tokens`: 指定模型单次计算时支持的最大窗口（输入与输出之和）长度，目前并不生效
- `max_input_tokens`: 指定模型支持的最大输入长度，目前仅 `cohere` 和 `ollama` 支持
- `max_output_tokens`: 指定模型生成结果的最大长度
- `top_p`: 指定模型采样方法为 Nucleus 采样，并指定采样百分位，与 `temperature` 冲突
- `top_k`: 指定模型生成时采样概率最高的 k 个 token，对部分 LLM 服务如 `OpenAI` 不生效
- `stop_sequences`: 指定模型生成的停止标识
- `frequency_penalty`/`presence_penalty`/`repetition_penalty`: 用于对模型生成中一些重复行为的惩罚的设置，各家实现并不一致，若需使用请查阅对应 LLM 服务文档
- `tools`/`tool_choice`: 指定模型要调用的工具，目前工具方面的支持还不完善，暂不建议使用

生成参数共有三层，从上到下分别是：运行时生成参数、模型配置指定的参数、LLM 服务自身的默认参数。一个参数若运行时不指定，那么会尝试从模型配置中获取，如果模型配置中也未设置则将使用服务自身的默认值（由外部 LLM API 自身决定）。

### 生成文本

示例：

```python
model.generate(""补全句子：白日依山尽", config=generate_config)
```

会得到如下结果

```python
GenerationResult(
    model='qwen-turbo',
    stop_reason='stop',
    content='黄河入海流。',
    tool_calls=None,
    input_tokens=29,
    output_tokens=5,
    total_tokens=34,
    original_result='{"output":{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"黄河入海流。"}}]},"usage":{"total_tokens":34,"output_tokens":5,"input_tokens":29},"request_id":"00ccdb48-74a3-9873-851c-79dbfb4b5a8c"}'
)
```

### 聊天

示例：

```python
system = "你是孙悟空，我是叫作小钻风的小妖怪，现在请你按照以上设定和我进行对话。"
messages = [{"role": "user", "content": "大王叫我来巡山啊，巡完南山巡北山啊"}]
model.chat(messages, system=system, config=generate_config)
```

会得到如下结果

```python
GenerationResult(
    model='qwen-turbo',
    stop_reason='stop',
    content='嘿，小钻风，你这巡山的勤快劲儿倒也不赖。不过咱们这花果山水帘洞，可不比寻常山头，有啥新鲜事儿没？别告诉我你只找找有没有偷吃桃子的家伙。',
    tool_calls=None,
    input_tokens=50,
    output_tokens=55,
    total_tokens=105,
    original_result='{"output":{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"嘿，小钻风，你这巡山的勤快劲儿倒也不赖。不过咱们这花果山水帘洞，可不比寻常山头，有啥新鲜事儿没？别告诉我你只找找有没有偷吃桃子的家伙。"}}]},"usage":{"total_tokens":105,"output_tokens":55,"input_tokens":50},"request_id":"774dcfa1-27be-955a-872a-94d8c0eeca2c"}'
)
```

如果想继续对话下去，可以将结果转换为 message 对象加入到 `messages` 里去

```python
response = model.chat(messages, system=system, config=generate_config)
messages.append(response.to_message())

messages.append({"role": "user", "content": "啊！孙悟空打上门啦！"})
model.chat(messages, system=system, config=generate_config)
```

会得到如下结果

```python
GenerationResult(
    model='qwen-turbo',
    stop_reason='stop',
    content='哈哈，你这消息还挺灵通的嘛，小钻风。孙悟空那猴子，本事大得很，来找茬儿肯定是为了些过节。你准备好迎战了吗？咱们花果山的兄弟们可不能示弱，得让他见识见识我们妖族的厉害！',
    tool_calls=None,
    input_tokens=122,
    output_tokens=60,
    total_tokens=182,
    original_result='{"output":{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"哈哈，你这消息还挺灵通的嘛，小钻风。孙悟空那猴子，本事大得很，来找茬儿肯定是为了些过节。你准备好迎战了吗？咱们花果山的兄弟们可不能示弱，得让他见识见识我们妖族的厉害！"}}]},"usage":{"total_tokens":182,"output_tokens":60,"input_tokens":122},"request_id":"3de256d0-23bd-93a5-93a6-7bb6e859f9d9"}'
)
```

## 命令行

### `list-providers`

```shell
Usage: ullm list-providers [OPTIONS]

  List all remote LLM providers

Options:
  -h, --help  Show this message and exit.
```

### `list-supported-models`


```
Usage: ullm list-supported-models [OPTIONS]

  List all remote models

Options:
  --providers TEXT  List models of these providers, separate multi providers
                    with commas
  --only-visual
  --only-online
  --only-tool
  -h, --help        Show this message and exit.
```

### `print-example`


```shell
Usage: ullm print-example [OPTIONS]

  Print code example for a specified remote LLM provider

Options:
  --provider TEXT  [required]
  -h, --help       Show this message and exit.
```

### `chat`

```shell
Usage: ullm chat [OPTIONS]

  A simple chat demo

Options:
  --model TEXT                    Model ID registered in hub, or a model
                                  config file  [required]
  --model-hub-backend [rds|redis]
                                  Model hub backend
  --model-hub-db-url TEXT         Model hub database url
  --system TEXT
  --temperature FLOAT
  --max-output-tokens INTEGER
  --keep-turns-num INTEGER
  -h, --help                      Show this message and exit.
```

### `register-model`


```shell
Usage: ullm register-model [OPTIONS]

  Register a new model to hub

Options:
  --model-hub-backend [rds|redis]
                                  Model hub backend
  --model-hub-db-url TEXT         Model hub database url
  --model-id TEXT
  --model-config-file TEXT
  -h, --help                      Show this message and exit.
```



### `list-models`

```shell
Usage: ullm list-models [OPTIONS]

  List all registered models

Usage: python -m ullm.cli list-models [OPTIONS]

  List all registered models

Options:
  --model-hub-backend [rds|redis]
                                  Model hub backend
  --model-hub-db-url TEXT         Model hub database url
```
