from __future__ import annotations
from typing import Dict, Any
from .base import VisionProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider

def get_provider(config_dict: Dict[str, Any]) -> VisionProvider:
    """
    Factory function to get the appropriate vision provider based on configuration.
    """
    provider_type = config_dict.get("api_provider", "ollama").lower()
    endpoint = config_dict.get("api_endpoint")
    
    if provider_type == "openai":
        if not endpoint:
            # Default OpenAI-compatible endpoint (e.g., llama.cpp default)
            endpoint = "http://localhost:8080/v1/chat/completions"
        return OpenAICompatProvider(endpoint)
    else:
        # Default to Ollama
        if not endpoint:
            endpoint = "http://localhost:11434/api/chat"
        return OllamaProvider(endpoint)
