"""Business rules. Unknown components receive no points and force review.

This yields a conservative evidence score, not an inferred quality estimate.
Semantic rejects require reviewer confirmation before payment is finalized.
"""

from qc.config import Config
from qc.schemas import Semantic, Technical, Visibility


def score(
    technical: Technical, visibility: Visibility, semantic: Semantic | None, cfg: Config
) -> dict:
    t, v, s = technical, visibility, semantic
    reasons = ["technical:" + key for key, passed in t.flags.items() if passed is False]
    notes = []
    technical_score = (
        sum(
            [
                t.blur_score,
                t.exposure_score,
                1 - t.frozen_frame_ratio,
                float(t.flags.get("resolution") is True),
                float(t.flags.get("fps") is True),
                float(t.flags.get("duration") is True),
            ]
        )
        / 6
    )
    hand = None
    obj = None
    task = None
    if s:
        task = (float(s.task_correct) + s.task_complete_score + float(s.final_state_success)) / 3
        hand = (s.hand_visibility_score + s.interaction_visibility_score) / 2
        obj = s.object_visibility_score
        if v.hand_visible_ratio is not None:
            hand = min(hand, v.hand_visible_ratio)
        if v.hand_object_visible_ratio is not None:
            hand = min(hand, v.hand_object_visible_ratio)
        if v.object_visible_ratio is not None:
            obj = min(obj, v.object_visible_ratio)
    framing = t.camera_stability_score
    if v.hand_boundary_ratio is not None:
        framing = (framing + 1 - v.hand_boundary_ratio) / 2
    if not t.available:
        technical_score = None
        framing = None
    components = dict(
        task=task, hand_interaction=hand, object=obj, technical=technical_score, framing=framing
    )
    quality = round(sum(cfg.weights[k] * (value or 0) for k, value in components.items()), 2)
    hard = []
    if t.corrupted:
        hard.append("corrupted_video")
        quality = 0.0
        components = {key: 0.0 for key in components}
        notes.append("Corruption overrides all component points and forces a zero score.")
    if s:
        if not s.task_correct:
            hard.append("wrong_task")
        if s.task_complete_score < cfg.incomplete_threshold:
            hard.append("task_clearly_incomplete")
        if s.interaction_visibility_score < cfg.critical_visibility_threshold:
            hard.append("critical_interaction_not_visible")
        if (
            s.irrelevant_footage_score >= cfg.irrelevant_threshold
            or s.task_complete_score <= cfg.no_manipulation_threshold
        ):
            hard.append("no_meaningful_manipulation")
        if s.protocol_violation:
            hard.append("severe_protocol_violation")
    unknown = (
        not t.available
        or s is None
        or v.hand_visible_ratio is None
        or v.object_visible_ratio is None
    )
    if not t.available:
        notes.append("Technical evaluation unavailable: " + (t.error or "unknown error"))
    if any(value is None for value in t.flags.values()):
        notes.append("Some technical checks are unavailable; inspect technical.flags.")
    if s is None:
        notes.append("Semantic evidence unavailable; unknown components receive zero points.")
    if v.hand_visible_ratio is None or v.object_visible_ratio is None:
        notes.append("Independent hand/object detector evidence unavailable.")
    decision = (
        "PASS"
        if quality >= cfg.pass_threshold
        else "REVIEW"
        if quality >= cfg.review_threshold
        else "FAIL"
    )
    borderline = cfg.borderline_margin > 0 and any(
        abs(quality - threshold) <= cfg.borderline_margin
        for threshold in (cfg.pass_threshold, cfg.review_threshold)
    )
    technical_failure = bool(reasons)
    disagreement = bool(
        s
        and (
            (
                v.hand_visible_ratio is not None
                and abs(s.hand_visibility_score - v.hand_visible_ratio) > cfg.disagreement_threshold
            )
            or (
                v.object_visible_ratio is not None
                and abs(s.object_visibility_score - v.object_visible_ratio)
                > cfg.disagreement_threshold
            )
        )
    )
    final_state_uncertain = s is not None and not s.final_state_success
    review = (
        unknown
        or borderline
        or (technical_failure and decision == "PASS")
        or disagreement
        or final_state_uncertain
        or bool(s and s.major_occlusion)
        or bool(v.offscreen_periods)
        or any(value is None for value in t.flags.values())
        or decision == "REVIEW"
    )
    if s and s.major_occlusion:
        reasons.append("major_occlusion")
    if v.offscreen_periods:
        reasons.append("possible_manipulation_offscreen")
    if final_state_uncertain:
        reasons.append("final_state_not_confirmed")
    if quality < cfg.review_threshold and not unknown:
        reasons.append("score_below_review_threshold")
    if review:
        decision = "REVIEW"
    if hard:
        decision = "FAIL"
        review = not t.corrupted  # A model judgment alone cannot finalize a pay rejection.
    if disagreement:
        notes.append(
            f"VLM and detector visibility differ by more than {cfg.disagreement_threshold}."
        )
    if borderline:
        notes.append("Score is within the configured decision-boundary review margin.")
    contributions = {
        key: round(cfg.weights[key] * (value or 0), 4) for key, value in components.items()
    }
    return dict(
        quality_score=quality,
        component_scores=components,
        weighted_contributions=contributions,
        decision=decision,
        failure_reasons=list(dict.fromkeys(hard + reasons)),
        human_review_required=review,
        evidence_notes=notes,
    )
