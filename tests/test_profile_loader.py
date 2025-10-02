from pathlib import Path

from cubiai.config.loader import InlineOverride, load_profile


def test_profile_loader_default(tmp_path: Path) -> None:
    profile = load_profile("anime-default")
    assert profile.name == "anime-default"
    assert profile.segmentation.num_segments > 0
    assert profile.export.psd.enabled is True
    assert profile.segmentation.backend == "huggingface-sam"
    assert profile.rigging.strategy == "llm"
    assert profile.rigging.builder is not None
    assert profile.rigging.builder.command


def test_inline_override_parsing() -> None:
    override = InlineOverride.parse("segmentation.num_segments=8")
    assert override.key_path == ("segmentation", "num_segments")
    assert override.value == 8
