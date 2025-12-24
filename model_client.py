import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

general_model_client = OpenAIChatCompletionClient(
    model = "qwen-plus-latest",
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = os.getenv("BASE_URL"),
    model_info = {
        "label": "qwen-plus-latest",
        "provider": "dashscope",
        "family": "qwen",
        "input_token_limit": 995904,
        "out_put_token_limit": 32768,
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "support_images": False,
        "is_reasoning_mopdel": True,
    },
    temperature=0.3,
)

coder_model_client = OpenAIChatCompletionClient(
    model = "qwen3-coder-plus",
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = os.getenv("BASE_URL"),
    model_info = {
        "label": "qwen3-coder-plus",
        "provider": "dashscope",
        "family": "qwen",
        "input_token_limit": 997952,
        "out_put_token_limit": 65536,
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "support_images": False,
        "is_reasoning_mopdel": True,
    },
    temperature=0.3,
)