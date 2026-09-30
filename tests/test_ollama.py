import json

import pytest

httpx = pytest.importorskip("httpx")
from qc.adapters.ollama_vlm import OllamaConfig, OllamaProvider  # noqa: E402
from qc.schemas import SemanticJudgment  # noqa: E402


def judgment():
    return dict(
        task_correct=False,
        task_complete_score=0.0,
        hand_visibility_score=0.9,
        object_visibility_score=0.0,
        interaction_visibility_score=0.0,
        final_state_success=False,
        major_occlusion=False,
        irrelevant_footage_score=1.0,
        protocol_violation=False,
        reasoning_summary="No requested manipulation observed.",
    )


def test_ollama_contract_and_metrics():
    seen = []

    def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": OllamaConfig().model, "digest": "sha256:weights"}]}
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test-runtime"})
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "message": {"content": json.dumps(judgment())},
                "eval_count": 25,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = OllamaProvider(client=client)
        answer = provider.evaluate(
            {
                "prompt": "Evaluate evidence",
                "metadata": {"expected_task_description": "move cup"},
                "frames": [
                    {"timestamp": 0, "jpeg_base64": "AA"},
                    {"timestamp": 1, "jpeg_base64": "BB"},
                ],
            }
        )
        assert SemanticJudgment.model_validate_json(answer).task_correct is False
        assert seen[0]["messages"][1]["images"] == ["AA", "BB"]
        assert seen[0]["options"]["temperature"] == 0
        assert "major_occlusion" in seen[0]["format"]["required"]
        assert provider.last_metrics["eval_count"] == 25


@pytest.mark.parametrize("bad", ["truncated", "invalid", "changed_digest"])
def test_ollama_rejects_unusable_evidence(bad):
    tag_calls = 0

    def handle(request):
        nonlocal tag_calls
        if request.url.path == "/api/tags":
            tag_calls += 1
            digest = "changed" if bad == "changed_digest" and tag_calls > 1 else "weights"
            return httpx.Response(
                200, json={"models": [{"name": OllamaConfig().model, "digest": digest}]}
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "length" if bad == "truncated" else "stop",
                "message": {"content": "{}"},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = OllamaProvider(client=client)
        with pytest.raises(ValueError):
            provider.evaluate(
                {"prompt": "p", "metadata": {}, "frames": [{"timestamp": 0, "jpeg_base64": "AA"}]}
            )
