"""Test local launcher path selection independently of actual inference."""

from pathlib import Path

from scripts import serve_local


def prepared_cache(root: Path, relative: str) -> Path:
    cache = root / relative
    snapshot = cache / (
        f"models--{serve_local.CHECKPOINT.replace('/', '--')}/snapshots/{serve_local.REVISION}"
    )
    snapshot.mkdir(parents=True)
    for name in serve_local.CHECKPOINT_FILES:
        (snapshot / name).write_bytes(b"path-selection fixture; not model weights")
    return cache


def test_local_launcher_preserves_explicit_setting(monkeypatch):
    monkeypatch.setenv("VLA_MODEL_CACHE", "/operator/selected/cache")
    serve_local.configure_environment()
    assert serve_local.os.environ["VLA_MODEL_CACHE"] == "/operator/selected/cache"


def test_local_launcher_prefers_project_cache(monkeypatch, tmp_path):
    root = tmp_path / "ECE525"
    monkeypatch.setattr(serve_local, "ROOT", root)
    monkeypatch.delenv("VLA_MODEL_CACHE", raising=False)
    project = prepared_cache(root, ".cache/huggingface/hub")
    prepared_cache(tmp_path, ".cache/huggingface/ece525/hub")
    serve_local.configure_environment()
    assert serve_local.os.environ["VLA_MODEL_CACHE"] == str(project)


def test_local_launcher_reuses_existing_cache_without_copy(monkeypatch, tmp_path):
    root = tmp_path / "ECE525"
    monkeypatch.setattr(serve_local, "ROOT", root)
    monkeypatch.delenv("VLA_MODEL_CACHE", raising=False)
    external = prepared_cache(tmp_path, ".cache/huggingface/ece525/hub")
    serve_local.configure_environment()
    assert serve_local.os.environ["VLA_MODEL_CACHE"] == str(external)
    assert not (root / ".cache").exists()


def test_local_launcher_missing_cache_keeps_portable_default(monkeypatch, tmp_path):
    monkeypatch.setattr(serve_local, "ROOT", tmp_path / "ECE525")
    monkeypatch.delenv("VLA_MODEL_CACHE", raising=False)
    serve_local.configure_environment()
    assert "VLA_MODEL_CACHE" not in serve_local.os.environ
