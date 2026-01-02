from __future__ import annotations
import requests
from .base import VisionProvider

class OpenAICompatProvider(VisionProvider):
    """
    OpenAI-compatible vision provider implementation.
    Supports llama.cpp, vLLM, LocalAI, etc.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def send_request(self, payload: dict, timeout: int) -> str:
        """
        Translates Ollama-style payload to OpenAI-compatible format and sends it.
        """
        # Translate Ollama payload to OpenAI format
        messages = []
        for msg in payload.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            images = msg.get("images", [])
            
            if not images:
                messages.append({"role": role, "content": content})
            else:
                # Build multi-modal content
                openai_content = [{"type": "text", "text": content}]
                for img_b64 in images:
                    # We assume JPEG as it's the default in fixxer's current implementation
                    openai_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    })
                messages.append({"role": role, "content": openai_content})

        openai_payload = {
            "model": payload.get("model"),
            "messages": messages,
            "stream": False
        }
        
        # Add response_format if requested (OpenAI style)
        if payload.get("format") == "json":
            openai_payload["response_format"] = {"type": "json_object"}

        response = requests.post(self.endpoint, json=openai_payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        
        # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
        return result['choices'][0]['message']['content']

    def check_connection(self) -> bool:
        """
        Checks connection by pinging the models endpoint.
        """
        base_url = self.endpoint.replace("/v1/chat/completions", "")
        models_url = f"{base_url}/v1/models"
        
        try:
            # Simple GET check
            response = requests.get(models_url, timeout=3)
            return response.status_code == 200
        except Exception:
            # Fallback: try to just hit the base URL
            try:
                response = requests.get(base_url, timeout=3)
                return response.status_code < 500
            except Exception:
                return False
