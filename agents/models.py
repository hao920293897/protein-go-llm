import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# vLLM configuration
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-xxx")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

deepseek_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENROUTER,
    model_type="deepseek/deepseek-chat-v3-0324:free",
    api_key=OPENROUTER_API_KEY,
    model_config_dict={"temperature": 0.3, "max_tokens": 140000},
)

gemini_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENROUTER,
    model_type="google/gemini-2.0-flash-001",
    api_key=OPENROUTER_API_KEY,
    model_config_dict={"temperature": 0.3, "max_tokens": 400000},
)

gpt_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENROUTER,
    model_type="openai/gpt-4.1-nano",
    api_key=OPENROUTER_API_KEY,
    model_config_dict={"temperature": 0.3, "max_tokens": 1000000},
)

# vLLM deployed Qwen model
qwen_vllm_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=VLLM_MODEL_NAME,
    api_key=VLLM_API_KEY,
    url=VLLM_API_BASE,
    model_config_dict={"temperature": 0.3, "max_tokens": 140000},
)

