from collections import Counter, defaultdict

import numpy as np

from qc.audit import compare
from qc.schemas import HumanReview, Result


def summarize(results: list[Result], reviews: list[HumanReview]) -> dict:
    # A video has one current automated evaluation in this dataset export.
    results = list({r.video_id: r for r in results}.values())
    collectors, tasks = defaultdict(list), defaultdict(list)
    for result in results:
        collectors[result.collector_id].append(result)
        tasks[result.task_name].append(result)

    def group(rows):
        return dict(
            count=len(rows),
            average_quality=float(np.mean([r.quality_score for r in rows])),
            failure_rate=sum(r.decision == "FAIL" for r in rows) / len(rows),
            review_rate=sum(r.human_review_required for r in rows) / len(rows),
        )

    def histogram(values, edges):
        counts, _ = np.histogram(values, bins=edges)
        return dict(bin_edges=edges, counts=counts.tolist(), missing=len(results) - len(values))

    return dict(
        clips=len(results),
        by_collector={k: group(v) for k, v in collectors.items()},
        by_task={k: group(v) for k, v in tasks.items()},
        common_failure_reasons=dict(
            Counter(reason for r in results for reason in r.failure_reasons)
        ),
        quality_distribution=histogram(
            [r.quality_score for r in results], [0, 20, 40, 60, 65, 75, 85, 90, 100]
        ),
        hand_visibility_distribution=histogram(
            [
                r.visibility.hand_visible_ratio
                for r in results
                if r.visibility.hand_visible_ratio is not None
            ],
            [0, 0.2, 0.4, 0.6, 0.8, 1],
        ),
        human_comparison=compare(results, reviews),
    )


def summary_rows(summary: dict) -> list[dict]:
    """Flatten aggregate data into labeled rows for CSV/JSONL dashboards."""
    rows = [dict(section="dataset", count=summary["clips"])]
    for section, source in (("collector", "by_collector"), ("task", "by_task")):
        rows.extend(
            dict(section=section, name=name, **values) for name, values in summary[source].items()
        )
    rows.extend(
        dict(section="failure_reason", name=name, count=count)
        for name, count in summary["common_failure_reasons"].items()
    )
    for section in ("quality_distribution", "hand_visibility_distribution"):
        histogram = summary[section]
        rows.extend(
            dict(section=section, lower=lower, upper=upper, count=count)
            for lower, upper, count in zip(
                histogram["bin_edges"], histogram["bin_edges"][1:], histogram["counts"]
            )
        )
        rows.append(dict(section=section + "_missing", count=histogram["missing"]))
    rows.append(dict(section="human_comparison", **summary["human_comparison"]))
    return rows
