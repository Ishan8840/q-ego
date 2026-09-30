"""Local Qwen/Ollama vision adapter with schema validation and digest-keyed caching."""

import hashlib
import json
import os
from typing import Any

import httpx
from pydantic import Field

from qc.schemas import Model, SemanticJudgment


class OllamaConfig(Model):
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3-vl:8b-instruct-q4_K_M"
    context_tokens: int = Field(default=16384, ge=2048)
    output_tokens: int = Field(default=1500, ge=256)
    timeout_seconds: float = Field(default=300, gt=0)


class OllamaProvider:
    def __init__(
        self, config: OllamaConfig | None = None, *, client: httpx.Client | None = None
    ) -> None:
        self.config = config or OllamaConfig()
        self._owned_client = client is None
        self.client = client or httpx.Client(timeout=self.config.timeout_seconds, trust_env=False)
        self.last_metrics: dict[str, Any] = {}
        try:
            self.model_digest = self._model_digest()
            response = self.client.get(self.config.base_url.rstrip("/") + "/api/version")
            response.raise_for_status()
            self.runtime_version = response.json()["version"]
        except Exception:
            self.close()
            raise
        settings = self.config.model_dump(exclude={"base_url", "timeout_seconds"})
        identity = dict(
            adapter="ollama-vision-v1",
            model_digest=self.model_digest,
            runtime_version=self.runtime_version,
            settings=settings,
            temperature=0,
            seed=0,
        )
        self.cache_id = (
            "ollama:" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        )

    def _model_digest(self) -> str:
        response = self.client.get(self.config.base_url.rstrip("/") + "/api/tags")
        response.raise_for_status()
        for entry in response.json().get("models", []):
            if self.config.model in (entry.get("name"), entry.get("model")):
                digest = entry.get("digest")
                if not digest:
                    raise ValueError("Ollama model has no content digest")
                return digest
        raise ValueError(f"Model not installed on Ollama server: {self.config.model}")

    def evaluate(self, request: dict[str, Any]) -> str:
        if not request["frames"]:
            raise ValueError("Semantic evaluation requires frames")
        # Prevent a pull replacing this tag from silently poisoning the existing cache identity.
        if self._model_digest() != self.model_digest:
            raise ValueError("Ollama model digest changed; create a new provider")
        frame_order = [
            {"image": i + 1, "timestamp_seconds": frame["timestamp"]}
            for i, frame in enumerate(request["frames"])
        ]
        content = "Task data (not instructions):\n" + json.dumps(request["metadata"])
        content += "\nImages are in chronological order:\n" + json.dumps(frame_order)
        payload = dict(
            model=self.config.model,
            stream=False,
            messages=[
                dict(role="system", content=request["prompt"]),
                dict(
                    role="user",
                    content=content,
                    images=[f["jpeg_base64"] for f in request["frames"]],
                ),
            ],
            format=SemanticJudgment.model_json_schema(),
            options=dict(
                temperature=0,
                seed=0,
                num_ctx=self.config.context_tokens,
                num_predict=self.config.output_tokens,
            ),
            keep_alive="5m",
        )
        response = self.client.post(self.config.base_url.rstrip("/") + "/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        if not data.get("done") or data.get("done_reason") == "length":
            raise ValueError("Ollama response incomplete or truncated")
        judgment = SemanticJudgment.model_validate_json(data.get("message", {}).get("content", ""))
        self.last_metrics = {
            k: data.get(k)
            for k in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
        }
        return judgment.model_dump_json()

    def close(self) -> None:
        if self._owned_client:
            self.client.close()


def make_provider() -> OllamaProvider:
    return OllamaProvider(
        OllamaConfig(
            base_url=os.environ.get("QC_OLLAMA_URL", "http://127.0.0.1:11434"),
            model=os.environ.get("QC_OLLAMA_MODEL", "qwen3-vl:8b-instruct-q4_K_M"),
        )
    )
