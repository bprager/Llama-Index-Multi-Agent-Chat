#!/usr/bin/env python3

import anthropic
from llama_index.llms.anthropic import Anthropic
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Settings
class ModelSettings(BaseSettings):
    """Settings for knowledge graph builder"""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )
    anthropic_api_key: SecretStr
    dial_api_key: SecretStr
    dial_url: str

settings = ModelSettings()

llm1 = Anthropic(model="claude-3-opus-20240229", api_key=settings.anthropic_api_key.get_secret_value())
llm2 = Anthropic(model="claude-3-opus-20240229", api_key=settings.dial_api_key.get_secret_value(), base_url=settings.dial_url)

resp = llm1.complete("Paul Graham is ")
print(resp)

try:
    resp = llm2.complete("Paul Graham is ")
except anthropic.AuthenticationError as e:
    print(f"\n** ERROR **: {e}")
else:
    print(resp)
