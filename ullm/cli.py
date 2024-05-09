import json
from operator import itemgetter

import click
from tabulate import tabulate

from .base import LanguageModel, RemoteLanguageModel


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


@main.command("list-models")
@click.option(
    "--providers", help="List models of these providers, separate multi providers with commas"
)
@click.option("--only-visual", is_flag=True)
@click.option("--only-online", is_flag=True)
@click.option("--only-tool", is_flag=True)
def list_models(providers, only_visual, only_online, only_tool):
    """List all remote models"""
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
@click.option("-c", "--config-file", required=True)
@click.option("--system")
@click.option("--temperature", type=float, default=0.7)
@click.option("--max-output-tokens", type=int, default=1024)
@click.option("--keep-turns-num", type=int, default=3)
def chat(config_file, system, temperature, max_output_tokens, keep_turns_num):
    """A simple chat demo"""
    config = None
    with open(config_file) as f:
        config = json.load(f)

    model = LanguageModel.from_config(config)
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


if __name__ == "__main__":
    main()
