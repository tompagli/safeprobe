"""OpenAI/Azure API helpers with retry logic."""
import os, time

def create_openai_client(api_key=None, azure_endpoint=None, azure_api_version="2024-02-15-preview"):
    from openai import OpenAI
    if azure_endpoint:
        from openai import AzureOpenAI
        return AzureOpenAI(azure_endpoint=azure_endpoint, api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"), api_version=azure_api_version)
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

def query_with_retry(client, model, messages, max_tokens=1024, temperature=0.0, max_retries=3, retry_delay=5.0):
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
            return r.choices[0].message.content
        except Exception as e:
            if attempt < max_retries-1: time.sleep(retry_delay*(attempt+1))
            else: raise
