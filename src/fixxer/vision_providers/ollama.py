from __future__ import annotations
import requests
from .base import VisionProvider

class OllamaProvider(VisionProvider):
    """
    Ollama-specific vision provider implementation.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def send_request(self, payload: dict, timeout: int) -> str:
        """
        Sends a request to Ollama's /api/chat endpoint.
        """
        response = requests.post(self.endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        
        # Ollama /api/chat format: {"message": {"content": "..."}}
        return result['message']['content']

    def check_connection(self) -> bool:
        """
        Checks Ollama connection using /api/tags (stripping /api/chat from endpoint if present).
        """
        base_url = self.endpoint.replace("/api/chat", "")
        tags_url = f"{base_url}/api/tags"
        
        try:
            response = requests.get(tags_url, timeout=3)
            return response.status_code == 200
        except Exception:
            return False
