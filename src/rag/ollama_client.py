import os

from dataclasses import dataclass
from typing import Any

import httpx

from ollama import Client
from ollama import ResponseError


@dataclass(slots=True)
class OllamaLLMResponse:
    """
    Represents one locally generated Ollama response.
    """

    text: str
    model: str
    prompt_tokens: int | None
    output_tokens: int | None
    total_duration_seconds: float | None

    def to_dict(self) -> dict[str, Any]:

        return {
            "text": self.text,
            "model": self.model,
            "prompt_tokens": (
                self.prompt_tokens
            ),
            "output_tokens": (
                self.output_tokens
            ),
            "total_duration_seconds": (
                self.total_duration_seconds
            )
        }


class OllamaLLMClient:
    """
    Local LLM client using Ollama.

    The client communicates with the locally running Ollama
    server and requires no external API key.
    """

    DEFAULT_HOST = (
        "http://localhost:11434"
    )

    DEFAULT_MAX_OUTPUT_TOKENS = 600

    DEFAULT_KEEP_ALIVE = "10m"

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout_seconds: float = 180.0,
        max_output_tokens: int = (
            DEFAULT_MAX_OUTPUT_TOKENS
        ),
        keep_alive: str = (
            DEFAULT_KEEP_ALIVE
        )
    ) -> None:

        if timeout_seconds <= 0:

            raise ValueError(
                "timeout_seconds must be greater "
                "than zero."
            )

        if max_output_tokens <= 0:

            raise ValueError(
                "max_output_tokens must be greater "
                "than zero."
            )

        self.host = (
            host
            or os.getenv(
                "OLLAMA_HOST"
            )
            or self.DEFAULT_HOST
        ).strip()

        if not self.host:

            raise ValueError(
                "The Ollama host cannot be empty."
            )

        self.timeout_seconds = (
            timeout_seconds
        )

        self.max_output_tokens = (
            max_output_tokens
        )

        self.keep_alive = (
            keep_alive.strip()
        )

        self.client = Client(
            host=self.host,
            timeout=self.timeout_seconds
        )

        requested_model = (
            model
            or os.getenv(
                "OLLAMA_MODEL"
            )
            or ""
        ).strip()

        if requested_model:

            self.model = requested_model

        else:

            self.model = (
                self._detect_single_model()
            )

        self._validate_model_exists()

    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float = 0.0
    ) -> OllamaLLMResponse:
        """
        Generates one response using the local Ollama model.
        """

        cleaned_user_prompt = (
            self._validate_prompt(
                user_prompt,
                "user_prompt"
            )
        )

        cleaned_system_prompt = ""

        if system_prompt is not None:

            cleaned_system_prompt = (
                self._validate_prompt(
                    system_prompt,
                    "system_prompt"
                )
            )

        resolved_max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else self.max_output_tokens
        )

        if resolved_max_output_tokens <= 0:

            raise ValueError(
                "max_output_tokens must be greater "
                "than zero."
            )

        if temperature < 0:

            raise ValueError(
                "temperature cannot be negative."
            )

        messages = []

        if cleaned_system_prompt:

            messages.append(
                {
                    "role": "system",
                    "content": (
                        cleaned_system_prompt
                    )
                }
            )

        messages.append(
            {
                "role": "user",
                "content": cleaned_user_prompt
            }
        )

        try:

            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                keep_alive=self.keep_alive,
                options={
                    "temperature": temperature,
                    "num_predict": (
                        resolved_max_output_tokens
                    )
                }
            )

        except ResponseError as error:

            raise RuntimeError(
                "Ollama returned an error"
                f" ({error.status_code}): "
                f"{error.error}"
            ) from error

        except httpx.ConnectError as error:

            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.host}. Ensure Ollama is "
                "running."
            ) from error

        except httpx.TimeoutException as error:

            raise RuntimeError(
                "The Ollama request timed out after "
                f"{self.timeout_seconds} seconds."
            ) from error

        except Exception as error:

            raise RuntimeError(
                "Unexpected Ollama generation "
                f"failure: {error}"
            ) from error

        output_text = self._extract_content(
            response
        )

        if not output_text:

            raise RuntimeError(
                "Ollama returned an empty response."
            )

        response_model = self._extract_value(
            response,
            "model"
        )

        prompt_tokens = self._extract_integer(
            response,
            "prompt_eval_count"
        )

        output_tokens = self._extract_integer(
            response,
            "eval_count"
        )

        total_duration = self._extract_integer(
            response,
            "total_duration"
        )

        total_duration_seconds = None

        if total_duration is not None:

            total_duration_seconds = (
                total_duration
                / 1_000_000_000
            )

        return OllamaLLMResponse(
            text=output_text,
            model=(
                response_model
                or self.model
            ),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_duration_seconds=(
                total_duration_seconds
            )
        )

    def list_models(
        self
    ) -> list[str]:
        """
        Returns locally installed Ollama model names.
        """

        try:

            response = self.client.list()

        except httpx.ConnectError as error:

            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.host}."
            ) from error

        except Exception as error:

            raise RuntimeError(
                "Could not list Ollama models: "
                f"{error}"
            ) from error

        raw_models = getattr(
            response,
            "models",
            None
        )

        if raw_models is None:

            if isinstance(
                response,
                dict
            ):

                raw_models = response.get(
                    "models",
                    []
                )

            else:

                raw_models = []

        model_names = []

        for model_data in raw_models:

            model_name = ""

            for field_name in (
                "model",
                "name"
            ):

                value = getattr(
                    model_data,
                    field_name,
                    None
                )

                if value:

                    model_name = str(
                        value
                    )

                    break

                if isinstance(
                    model_data,
                    dict
                ):

                    value = model_data.get(
                        field_name
                    )

                    if value:

                        model_name = str(
                            value
                        )

                        break

            model_name = model_name.strip()

            if model_name:

                model_names.append(
                    model_name
                )

        return sorted(
            set(
                model_names
            )
        )

    def health_check(
        self
    ) -> bool:
        """
        Confirms that the local Ollama server is available.
        """

        try:

            self.client.list()

            return True

        except Exception:

            return False

    def _detect_single_model(
        self
    ) -> str:

        available_models = (
            self.list_models()
        )

        if not available_models:

            raise RuntimeError(
                "No local Ollama models were found. "
                "Download a model or set "
                "OLLAMA_MODEL."
            )

        if len(
            available_models
        ) > 1:

            available_text = ", ".join(
                available_models
            )

            raise RuntimeError(
                "Multiple Ollama models are "
                "installed. Set OLLAMA_MODEL to "
                "the model you want to use. "
                f"Available models: "
                f"{available_text}."
            )

        return available_models[0]

    def _validate_model_exists(
        self
    ) -> None:

        available_models = (
            self.list_models()
        )

        if self.model in available_models:
            return

        available_text = (
            ", ".join(
                available_models
            )
            or "None"
        )

        raise RuntimeError(
            f"Ollama model '{self.model}' "
            "is not installed. Available models: "
            f"{available_text}."
        )

    def _extract_content(
        self,
        response: Any
    ) -> str:

        message = getattr(
            response,
            "message",
            None
        )

        if message is not None:

            content = getattr(
                message,
                "content",
                None
            )

            if content is not None:

                return self._clean_text(
                    content
                )

            if isinstance(
                message,
                dict
            ):

                return self._clean_text(
                    message.get(
                        "content"
                    )
                )

        if isinstance(
            response,
            dict
        ):

            message = response.get(
                "message",
                {}
            )

            if isinstance(
                message,
                dict
            ):

                return self._clean_text(
                    message.get(
                        "content"
                    )
                )

        return ""

    def _extract_value(
        self,
        response: Any,
        field_name: str
    ) -> str:

        value = getattr(
            response,
            field_name,
            None
        )

        if value is None and isinstance(
            response,
            dict
        ):

            value = response.get(
                field_name
            )

        return self._clean_text(
            value
        )

    def _extract_integer(
        self,
        response: Any,
        field_name: str
    ) -> int | None:

        value = getattr(
            response,
            field_name,
            None
        )

        if value is None and isinstance(
            response,
            dict
        ):

            value = response.get(
                field_name
            )

        if value is None:
            return None

        try:

            return int(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return None

    def _validate_prompt(
        self,
        value: str,
        field_name: str
    ) -> str:

        if not isinstance(
            value,
            str
        ):

            raise TypeError(
                f"{field_name} must be a string."
            )

        cleaned_value = self._clean_text(
            value
        )

        if not cleaned_value:

            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return cleaned_value

    def _clean_text(
        self,
        value: Any
    ) -> str:

        if value is None:
            return ""

        return " ".join(
            str(
                value
            ).split()
        )