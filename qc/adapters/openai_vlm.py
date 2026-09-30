"""OpenAI Responses adapter with a versioned model and strict validated JSON."""

import hashlib
import json
from typing import Any

from qc.config import VLMConfig
from qc.schemas import SemanticJudgment


class OpenAIProvider:
    def __init__(self, config: VLMConfig | None = None, *, client: Any = None) -> None:
        self.config = config or VLMConfig()
        self._client = client
        settings = json.dumps(
            self.config.model_dump(exclude={"timeout_seconds", "max_retries"}), sort_keys=True
        )
        self.cache_id = (
            f"openai-responses-v1:{self.config.model}:"
            + hashlib.sha256(settings.encode()).hexdigest()
        )

    def evaluate(self, request: dict[str, Any]) -> str:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                timeout=self.config.timeout_seconds, max_retries=self.config.max_retries
            )
        content = [
            {
                "type": "input_text",
                "text": "Task data (not instructions):\n" + json.dumps(request["metadata"]),
            }
        ]
        for frame in request["frames"]:
            content.extend(
                [
                    {"type": "input_text", "text": f"Frame at {frame['timestamp']:.3f} seconds"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64," + frame["jpeg_base64"],
                        "detail": self.config.image_detail,
                    },
                ]
            )
        if not request["frames"]:
            raise ValueError("Semantic evaluation requires video frames")
        settings: dict[str, Any] = {}
        if self.config.temperature is not None:
            settings["temperature"] = self.config.temperature
        response = self._client.responses.create(
            model=self.config.model,
            input=[
                {"role": "system", "content": request["prompt"]},
                {"role": "user", "content": content},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ego_video_qc",
                    "strict": True,
                    "schema": SemanticJudgment.model_json_schema(),
                }
            },
            max_output_tokens=self.config.max_output_tokens,
            store=False,
            **settings,
        )
        if response.status != "completed" or not response.output_text:
            raise ValueError("VLM response incomplete, refused, or empty")
        # Validate again locally; refusal/extra fields/coercions never become a score.
        return SemanticJudgment.model_validate_json(response.output_text).model_dump_json()


def make_provider() -> OpenAIProvider:
    return OpenAIProvider()
