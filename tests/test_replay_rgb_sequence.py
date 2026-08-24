"""Input selection tests for deterministic RGB mapping replay."""

from scripts.diagnostics.replay_rgb_sequence import list_rgb_frames


def test_replay_excludes_semantic_and_rendered_images(tmp_path):
    for name in (
            "rgb_000002.jpg", "rgb_000001.PNG", "topk_frame_000001.png",
            "pointing_000001.jpg", "rgb_preview.jpg", "notes.json"):
        (tmp_path / name).touch()

    assert [path.name for path in list_rgb_frames(tmp_path)] == [
        "rgb_000001.PNG", "rgb_000002.jpg"]
