import json
import os
from operator import itemgetter

import arrow
import click
from tabulate import tabulate

from .base import LanguageModel, RemoteLanguageModel
from .hub import ModelHub


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    pass


@main.command("list-providers")
def list_providers():
    """List all remote LLM providers"""
    providers = RemoteLanguageModel.list_providers()
    headers = ["name", "models", "visual_models", "tool_models", "online_models"]
    data = [itemgetter(*headers)(provider) for provider in providers]
    print(tabulate(data, headers=headers, tablefmt="github"))


@main.command("list-supported-models")
@click.option(
    "--providers", help="List models of these providers, separate multi providers with commas"
)
@click.option("--only-visual", is_flag=True)
@click.option("--only-online", is_flag=True)
@click.option("--only-tool", is_flag=True)
def list_supported_models(providers, only_visual, only_online, only_tool):
    """List all supported remote models"""
    providers = None if not providers else set(providers.split(","))
    models = RemoteLanguageModel.list_models(
        providers=providers, visual=only_visual, online=only_online, tools_enable=only_tool
    )
    headers = ["provider", "model", "visual", "online", "tools"]
    data = [itemgetter(*headers)(model_info) for model_info in models]
    print(tabulate(data, headers=headers, tablefmt="github"))


@main.command("print-example")
@click.option("--provider", required=True)
def print_example(provider):
    """Print code example for a specified remote LLM provider"""
    example_template = """Example code:\n```\nfrom ullm import LanguageModel

model_config = {config}
model = LanguageModel.from_config(model_config)
messages = [{{"role": "user", "content": "Hello!"}}]
res = model.chat(messages)
messages.append(res.to_message())
messages.append({{"role": "user", "content": "Tell me a joke please!"}})
res = model.chat(messages)
```"""
    required_config, optional_config = RemoteLanguageModel.get_provider_example(provider)
    config = "{\n"
    config += "    # required fields\n"
    for key, value in required_config.items():
        config += f'    "{key}": {repr(value)},\n'

    config += "\n    # optional fields\n"
    for key, value in optional_config.items():
        config += f'    "{key}": {repr(value)},\n'

    config += "}"
    print(example_template.format(config=config))


@main.command("chat")
@click.option("--model", help="Model ID registered in hub, or a model config file", required=True)
@click.option("--model-hub-db-url", help="Model hub database url")
@click.option("--system")
@click.option("--temperature", type=float, default=0.7)
@click.option("--max-output-tokens", type=int, default=1024)
@click.option("--keep-turns-num", type=int, default=3)
def chat(model, model_hub_db_url, system, temperature, max_output_tokens, keep_turns_num):
    """A simple chat demo"""
    model_id_or_config_file = model
    if os.path.exists(model_id_or_config_file):
        config, config_file = None, model_id_or_config_file
        with open(config_file) as f:
            config = json.load(f)

        model = LanguageModel.from_config(config)
    else:
        model = ModelHub(model_hub_db_url).get_model(model_id_or_config_file)

    if not model:
        click.secho(
            f"`{model_id_or_config_file}` is not a valid model id or a valid model config file"
        )
        return -1

    generate_config = {"temperature": temperature, "max_output_tokens": max_output_tokens}
    messages = []
    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue

        if user_message.lower() in ("exit", "quit"):
            break

        messages = messages[-keep_turns_num * 2 :] + [{"role": "user", "content": user_message}]
        response = model.chat(messages, generate_config)
        print(f"Bot: {response.content}\n")
        messages.append(response.to_message())


@main.command("register-model")
@click.option("--db-url", help="Model hub database url")
@click.option("--model-id", required=True)
@click.option("--model-config-file", required=True)
def register_model(db_url, model_id, model_config_file):
    "Register a new model to hub"
    hub = ModelHub(db_url)
    model_config = json.load(open(model_config_file))
    model = LanguageModel.from_config(model_config)
    hub.register_model(model, model_id)


@main.command("list-models")
@click.option("--db-url", help="Model hub database url")
def list_models(db_url):
    """List all registered models"""
    hub = ModelHub(db_url)
    headers = ["Model ID", "Model Name", "Remote", "Created"]
    data = []
    for model_record in hub.list_models():
        data.append(
            [
                model_record.model_id,
                model_record.model_name,
                model_record.remote,
                str(arrow.get(model_record.created, tzinfo="UTC")),
            ]
        )

    print(tabulate(data, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
