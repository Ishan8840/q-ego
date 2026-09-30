import pytest

from scripts.egodex_smoke import description, select_members


def test_egodex_reversible_description_direction():
    assert description({"llm_description": b"open lid"}) == "open lid"
    assert (
        description({"llm_description": "fold shirt", "llm_description2": "None"}) == "fold shirt"
    )
    assert (
        description(
            {
                "llm_description": "open lid",
                "llm_description2": "close lid",
                "which_llm_description": 2,
            }
        )
        == "close lid"
    )
    with pytest.raises(ValueError, match="selector"):
        description({"llm_description": "open", "llm_description2": "close"})
    with pytest.raises(ValueError, match="selector"):
        description({"llm_description": "open", "which_llm_description": 0})
    with pytest.raises(ValueError, match="Missing"):
        description({})


def test_egodex_selection_is_reproducible_paired_and_safe():
    members = [
        f"test/{task}/{i}.{ext}"
        for task in ["a", "b", "c"]
        for i in range(4)
        for ext in ["mp4", "hdf5"]
    ]
    members += ["test/a/unpaired.mp4", "../escape/x.mp4", "../escape/x.hdf5"]
    first = select_members(members, 2, 3, "seed")
    assert first == select_members(list(reversed(members)), 2, 3, "seed")
    assert len(first) == 6
    assert len({name.split("/")[1] for name in first}) == 2
    assert not any("unpaired" in name or ".." in name for name in first)
    with pytest.raises(ValueError):
        select_members(members, 4, 3, "seed")
