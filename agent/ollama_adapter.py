import os
from typing import Optional

try:
    from ollama import Client, Options, GenerateResponse
except ImportError as e:
    raise RuntimeError(
        "The 'ollama' Python client is required for local Ollama usage.\n"
        "Install it with: pip install ollama\n"
        f"Underlying error: {e}"
    )


class OllamaAdapter:
    """Minimal adapter around the `ollama` Python client.

    Provides a `generate(prompt: str) -> str` method returning model text.
    The adapter is intentionally defensive about client API differences
    across versions.
    """
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        try:
            # The python package name is `ollama` (must be installed)
            from ollama import Client, Options, GenerateResponse
        except Exception as e:
            raise RuntimeError(
                "The 'ollama' Python client is required for local Ollama usage.\n"
                "Install it with: pip install ollama\n"
                f"Underlying error: {e}"
            )

        # Create client; Client() typically connects to the local Ollama daemon.
        if base_url:
            self.client = Client(base_url=base_url)
        else:
            self.client = Client()

        # Default model name
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma3:1b")

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
        """Generate text from the configured local Ollama model using ollama.Client v0.1+ API."""
        try:
            res = self.client.generate(
                model=self.model,
                prompt=prompt,
                options=Options(
                    temperature=temperature,
                    num_predict=max_tokens
                ),
                stream=False
            )
            if hasattr(res, '__iter__'):
                # If iterator, consume and get last response
                full_response = ''
                for chunk in res:
                    full_response += chunk['response']
                return full_response
            else:
                return res['response']
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
