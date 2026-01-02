from __future__ import annotations
from abc import ABC, abstractmethod

class VisionProvider(ABC):
    """
    Abstract base class for vision providers.
    """

    @abstractmethod
    def send_request(self, payload: dict, timeout: int) -> str:
        """
        Sends a request to the AI model and returns the raw text content.

        Args:
            payload: The logical intent of the request (messages, model, etc.)
            timeout: Request timeout in seconds

        Returns:
            The raw text content from the model's response.
        """
        pass

    @abstractmethod
    def check_connection(self) -> bool:
        """
        Checks if the provider is running and accessible.

        Returns:
            True if accessible, False otherwise.
        """
        pass
