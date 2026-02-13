"""
safeprobe.datasets.adapters - Unified interface for LLM backends.

Implements the adapter pattern enabling SafeProbe to evaluate diverse models
across different deployment configurations (FR1) without requiring changes
to the attack or analysis logic.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class LLMAdapter(ABC):
    """Abstract base class for LLM backend adapters."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def query(self, messages: List[Dict[str, str]], temperature: float = 0.0,
              max_tokens: int = 1024) -> str:
        """Send a query to the LLM and return the response text."""
        pass

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0,
             max_tokens: int = 1024) -> str:
        """Convenience method for single-turn chat."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.query(messages, temperature=temperature, max_tokens=max_tokens)


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API (including Azure OpenAI)."""

    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def query(self, messages: List[Dict[str, str]], temperature: float = 0.0,
              max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AzureOpenAIAdapter(LLMAdapter):
    """Adapter for Azure OpenAI API."""

    def __init__(self, model_name: str, azure_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None, api_version: str = "2024-02-15-preview",
                 **kwargs):
        super().__init__(model_name, **kwargs)
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=api_version,
        )

    def query(self, messages: List[Dict[str, str]], temperature: float = 0.0,
              max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        import anthropic
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    def query(self, messages: List[Dict[str, str]], temperature: float = 0.0,
              max_tokens: int = 1024) -> str:
        # Extract system prompt if present
        system = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=user_messages,
            system=system,
            temperature=temperature,
        )
        return response.content[0].text


class OllamaAdapter(LLMAdapter):
    """Adapter for local Ollama models."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

    def query(self, messages: List[Dict[str, str]], temperature: float = 0.0,
              max_tokens: int = 1024) -> str:
        import ollama
        response = ollama.chat(model=self.model_name, messages=messages)
        return response["message"]["content"]


class XAIAdapter(LLMAdapter):
    """Adapter for xAI Grok API."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )

    def query(self, messages: List[Dict[str, str]], temperature: float = 0.0,
              max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def create_adapter(model_name: str, model_type: str, **kwargs) -> LLMAdapter:
    """
    Factory function to create the appropriate adapter.

    Args:
        model_name: Model identifier (e.g., 'gpt-4.1', 'claude-3-opus-20240229')
        model_type: Backend type ('openai', 'azure', 'anthropic', 'ollama', 'xai', 'google')
        **kwargs: Additional arguments passed to the adapter constructor.

    Returns:
        An initialized LLMAdapter instance.
    """
    adapters = {
        "openai": OpenAIAdapter,
        "azure": AzureOpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "ollama": OllamaAdapter,
        "xai": XAIAdapter,
    }

    if model_type not in adapters:
        raise ValueError(
            f"Unsupported model type: '{model_type}'. "
            f"Supported types: {list(adapters.keys())}"
        )

    return adapters[model_type](model_name, **kwargs)

# Alias with api_key support
def get_adapter(model_name: str, model_type: str, api_key=None, **kwargs) -> LLMAdapter:
    if api_key:
        kwargs["api_key"] = api_key
    return create_adapter(model_name, model_type, **kwargs)

