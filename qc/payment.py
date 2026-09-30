from qc.config import Config


def recommend(score: float, decision: str, review: bool, cfg: Config) -> tuple[float, str]:
    if review:
        return 0.0, "held_for_review"
    if decision == "FAIL" or score < cfg.reduced_pay_threshold:
        return 0.0, "rejected"
    if score >= cfg.bonus_threshold:
        return cfg.bonus_multiplier, "recommended"
    if score >= cfg.full_pay_threshold:
        return 1.0, "recommended"
    return cfg.reduced_multiplier, "recommended"
