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
        # Newer OpenAI models (gpt-5, o-series) require max_completion_tokens;
        # fall back to max_tokens for older models that don't accept the new param.
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        except Exception as e:
            if "max_completion_tokens" in str(e) or "unsupported_parameter" in str(e):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise
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
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        except Exception as e:
            if "max_completion_tokens" in str(e) or "unsupported_parameter" in str(e):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise
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


class HuggingFaceAdapter(LLMAdapter):
    """Adapter for any HuggingFace causal instruct model.

    Supports Llama-3, Mistral, Qwen3, and any other model with a
    built-in chat template in its tokenizer.

    Recommended open-source targets:
      - meta-llama/Llama-3.2-3B-Instruct
      - mistralai/Mistral-7B-Instruct-v0.3
      - Qwen/Qwen3-4B

    Args:
        model_name:   HuggingFace model ID.
        device_map:   Passed to from_pretrained (default "auto").
        load_in_4bit: Use 4-bit NF4 quantization via bitsandbytes.
        hf_token:     HuggingFace access token for gated models.
        enable_thinking: For Qwen3 models — pass False to suppress
                         <think> blocks (default True).
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        load_in_4bit: bool = False,
        hf_token: Optional[str] = None,
        enable_thinking: bool = True,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._enable_thinking = enable_thinking
        tok_kwargs: Dict = {"trust_remote_code": True}
        if hf_token:
            tok_kwargs["token"] = hf_token

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: Dict = {"trust_remote_code": True}
        if hf_token:
            model_kwargs["token"] = hf_token

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["device_map"] = device_map
        elif torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = device_map
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if not torch.cuda.is_available() and not load_in_4bit:
            self._model = self._model.to("cpu")

    def _device(self):
        """Return the model's device without triggering StopIteration from device_map."""
        import torch
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _apply_template(self, messages: List[Dict[str, str]]) -> "torch.Tensor":
        """Run apply_chat_template and always return a plain input_ids tensor.

        Transformers ≥5 returns a BatchEncoding; ≤4 returns a plain tensor.
        Both cases are handled here so callers never see a BatchEncoding.
        """
        template_kwargs: Dict = {"add_generation_prompt": True, "return_tensors": "pt"}
        device = self._device()
        try:
            result = self._tokenizer.apply_chat_template(
                messages,
                enable_thinking=self._enable_thinking,
                **template_kwargs,
            )
        except TypeError:
            # Tokenizer does not support enable_thinking — ignore it
            result = self._tokenizer.apply_chat_template(messages, **template_kwargs)

        # BatchEncoding (transformers ≥5) vs plain tensor (transformers ≤4)
        if hasattr(result, "input_ids"):
            return result.input_ids.to(device)
        return result.to(device)

    def query(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        import torch

        input_ids = self._apply_template(messages)

        eos_id = self._tokenizer.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        pad_id = self._tokenizer.pad_token_id or eos_id or 0

        gen_kwargs: Dict = {
            "max_new_tokens": max_tokens,
            "pad_token_id": pad_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output = self._model.generate(input_ids=input_ids, **gen_kwargs)

        new_tokens = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
        "huggingface": HuggingFaceAdapter,
        "hf": HuggingFaceAdapter,
        "local": HuggingFaceAdapter,
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

