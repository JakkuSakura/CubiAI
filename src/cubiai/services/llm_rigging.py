"""LLM-backed rigging assistant."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import httpx

from ..pipeline.artifacts import LayerArtifact


class RiggingLLMError(RuntimeError):
    """Raised when the LLM response cannot be parsed or fails."""


@dataclass(slots=True)
class LLMRiggingClient:
    """Client for OpenAI-compatible chat completion endpoints used for rig generation."""

    model: str
    base_url: str
    api_key: str
    timeout_seconds: int = 120

    def generate_rig_configuration(self, layers: Iterable[LayerArtifact]) -> dict[str, object]:
        layer_summaries = [
            {
                "name": layer.name,
                "bbox": layer.bbox,
                "metadata": layer.metadata,
            }
            for layer in layers
        ]

        system_prompt = (
            "You are a Live2D rigging assistant. Given layer metadata, respond with JSON defining "
            "`parts`, `parameters`, `deformers`, and `motions` arrays suitable for an automatic rig."
        )
        user_prompt = (
            "Create a Live2D rig configuration for the following layers. "
            "Respond with a JSON object containing keys parts, parameters, deformers, physics, and motions. "
            "Each part should include id, layer, bounds, pivot, and blendShape (if applicable). "
            "Parameters should define realistic min/max/default values for Cubism parameters. "
            "Motions should include idle and blink behaviours.\n\n"
            + json.dumps({"layers": layer_summaries})
        )

        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        try:
            response = httpx.post(endpoint, headers=headers, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - network dependent
            detail = exc.response.text
            raise RiggingLLMError(f"LLM request failed: {detail}") from exc
        except httpx.HTTPError as exc:  # pragma: no cover - network dependent
            raise RiggingLLMError(str(exc)) from exc

        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RiggingLLMError("Unexpected response schema from LLM provider") from exc

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RiggingLLMError("LLM did not return valid JSON rig configuration") from exc

        mandatory_keys = {"parts", "parameters", "motions"}
        missing = [key for key in mandatory_keys if key not in parsed]
        if missing:
            raise RiggingLLMError(f"LLM response missing keys: {', '.join(missing)}")

        return parsed
