# SPDX-License-Identifier: GPL-3.0-or-later
"""Local checkpoint storage. Only explicit download jobs may contact the Hub.

No Blender, torch, Kimodo, or Hugging Face imports at module scope: the worker
must configure its environment before those libraries cache environment values.
"""
import json
import os
from pathlib import Path
import re

MODEL_NAMES = (
    "Kimodo-SOMA-RP-v1.1", "Kimodo-SOMA-SEED-v1.1",
    "Kimodo-SOMA-RP-v1", "Kimodo-SOMA-SEED-v1",
    "Kimodo-G1-RP-v1", "Kimodo-G1-SEED-v1", "Kimodo-SMPLX-RP-v1",
)
TEXT_REPOS = (
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
)
DOWNLOAD_OPERATIONS = {"download_model", "download_text"}
STORAGE_OPERATIONS = DOWNLOAD_OPERATIONS | {"check_models"}
STATE_FILE = ".kimodo-download.json"


def repository_id(model):
    """Resolve shipped preset names/short keys, or accept an explicit Hub ID."""
    aliases = {name.lower(): "nvidia/" + name for name in MODEL_NAMES}
    aliases.update({"kimodo-soma-rp": "nvidia/Kimodo-SOMA-RP-v1.1",
                    "kimodo-soma-seed": "nvidia/Kimodo-SOMA-SEED-v1.1",
                    "kimodo-g1-rp": "nvidia/Kimodo-G1-RP-v1",
                    "kimodo-g1-seed": "nvidia/Kimodo-G1-SEED-v1",
                    "kimodo-smplx-rp": "nvidia/Kimodo-SMPLX-RP-v1"})
    value = str(model).strip()
    if value.lower() in aliases:
        return aliases[value.lower()]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("Choose a model preset or an explicit owner/model Hub identifier")
    if ".." in value or "--" in value:
        raise ValueError("Invalid model identifier")
    return value


def absolute_folder(value, label):
    if not value or not str(value).strip():
        raise ValueError(f"Choose {label} first")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute local folder path")
    return path.resolve()


def cache_home(storage):
    if storage.get("cache_root"):
        return absolute_folder(storage["cache_root"], "the dependency cache folder")
    if storage.get("models_root"):
        return absolute_folder(storage["models_root"], "the model download folder") / ".huggingface"
    return None


def checkpoint_folder(storage, model, *, download=False):
    mode = storage.get("mode", "MANAGED")
    if mode not in {"MANAGED", "MANUAL"}:
        raise ValueError("Unknown model storage mode")
    if mode == "MANUAL":
        if download:
            raise ValueError("Manual model folders are read-only. Switch to Download folder to download")
        return absolute_folder(storage.get("manual_path"), "the manual checkpoint folder")
    root = absolute_folder(storage.get("models_root"), "the model download folder")
    owner, name = repository_id(model).split("/")
    # Keep official folder names readable; isolate custom owners from NVIDIA.
    folder = root / (name if owner == "nvidia" else owner + "__" + name)
    if folder.resolve().parent != root:
        raise ValueError("Model destination must not escape the chosen folder through a symlink")
    return folder


def configure_environment(request):
    """Call before importing Kimodo/HF. Changes apply only to this worker process."""
    storage = request.get("storage", {})
    download = request.get("operation") in DOWNLOAD_OPERATIONS
    if download and request.get("download_confirmed") is not True:
        raise ValueError("Downloads require an explicit Download button confirmation")
    home = cache_home(storage)
    if download and home is None:
        raise ValueError("Choose a model download folder or dependency cache folder first")
    if home is not None:
        # Retain the existing login when relocating model storage. Never serialize tokens.
        old_home = Path(os.environ.get("HF_HOME") or
                        str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home()/".cache"))) / "huggingface"))
        os.environ.setdefault("HF_TOKEN_PATH", str(old_home / "token"))
        os.environ.update(HF_HOME=str(home), HF_HUB_CACHE=str(home / "hub"),
                          HUGGINGFACE_HUB_CACHE=str(home / "hub"),
                          HUGGINGFACE_CACHE_DIR=str(home / "hub"),
                          TRANSFORMERS_CACHE=str(home / "hub"), HF_XET_CACHE=str(home / "xet"))
    os.environ["HF_HUB_OFFLINE"] = "0" if download else "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0" if download else "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TEXT_ENCODER_MODE"] = "local"  # Never probe a remote/API text encoder.
    if storage.get("text_encoder_root"):
        os.environ["TEXT_ENCODERS_DIR"] = str(absolute_folder(storage["text_encoder_root"], "the text encoder root"))
    elif "storage" in request:
        os.environ.pop("TEXT_ENCODERS_DIR", None)
    if storage.get("text_encoder_device"):
        os.environ["TEXT_ENCODER_DEVICE"] = storage["text_encoder_device"]


def _json(path):
    if path.stat().st_size > 4_000_000:
        raise ValueError(f"Metadata file is too large: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    with path.open("rb") as f:
        if f.read(64).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Git LFS pointer is not model data: {path}")


def checkpoint_config(folder, *, ignore_pending=False):
    """Parse without instantiating; check referenced weights and normalization stats."""
    from omegaconf import OmegaConf
    folder = Path(folder).resolve()
    state = folder / STATE_FILE
    if state.exists() and not ignore_pending and _json(state).get("state") != "complete":
        raise ValueError("An interrupted download is present. Click Download / resume before generation")
    config = folder / "config.yaml"
    _file(config)
    if config.stat().st_size > 2_000_000:
        raise ValueError("Checkpoint config.yaml is too large")
    conf = OmegaConf.load(config)
    if not OmegaConf.is_dict(conf):
        raise ValueError("config.yaml must contain a Kimodo model mapping")
    # Resolve standard checkpoint-relative interpolation without loading model code.
    merged = OmegaConf.merge(conf, {"checkpoint_dir": str(folder), "text_encoder": None})
    resolved = OmegaConf.to_container(merged, resolve=True)
    if not str(resolved.get("_target_", "")).startswith("kimodo."):
        raise ValueError("config.yaml is not a Kimodo checkpoint configuration")
    checked = []
    def walk(value):
        if isinstance(value, dict):
            for v in value.values(): walk(v)
        elif isinstance(value, list):
            for v in value: walk(v)
        elif isinstance(value, str) and value.replace("\\", "/").startswith(str(folder).replace("\\", "/").rstrip("/") + "/"):
            path = Path(value)
            if not path.exists():
                raise ValueError(f"Missing checkpoint asset: {path}")
            if path.is_file():
                _file(path); checked.append(path)
            else:
                # Normalizer paths contain mean/std arrays. Detect partial snapshots.
                groups = [path / x for x in ("body", "global_root", "local_root")] if path.name == "motion" else [path]
                for group in groups:
                    for name in ("mean.npy", "std.npy"):
                        _file(group / name); checked.append(group / name)
    walk(resolved)
    weights = [p for p in checked if p.suffix in {".safetensors", ".pt", ".pth", ".ckpt", ".bin"}]
    if not weights:
        _file(folder / "model.safetensors")
        checked.append(folder / "model.safetensors")
    return conf, checked


def validate_text_snapshot(folder, repo):
    folder = Path(folder)
    if repo == TEXT_REPOS[0]:
        _file(folder / "config.json")
        _file(folder / "tokenizer_config.json")
        _file(folder / "tokenizer.json")
        indices = list(folder.glob("*.index.json"))
        if indices:
            shards = set()
            for index in indices:
                shards.update(_json(index).get("weight_map", {}).values())
            if not shards:
                raise ValueError("Text encoder weight index has no shards")
            for name in shards:
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe text encoder shard path")
                _file(folder / relative)
        else:
            weight = folder / "model.safetensors"
            if not weight.exists(): weight = folder / "pytorch_model.bin"
            _file(weight)
    else:
        _file(folder / "adapter_config.json")
        _file(folder / "adapter_model.safetensors")
        if repo == TEXT_REPOS[1]:
            _file(folder / "config.json")
            _file(folder / "tokenizer.json")
            _file(folder / "tokenizer_config.json")


def inspect_storage(request):
    from huggingface_hub import snapshot_download
    storage = request.get("storage", {})
    rows = []
    try:
        folder = checkpoint_folder(storage, request["model"])
        _, checked = checkpoint_config(folder)
        rows.append({"component": "Motion model", "ready": True, "path": str(folder),
                     "message": f"Configuration and {len(checked)} referenced files present"})
    except Exception as exc:
        rows.append({"component": "Motion model", "ready": False, "message": str(exc)})
    home = cache_home(storage)
    hub = str(home / "hub") if home else None
    for repo in TEXT_REPOS:
        try:
            manual_root = storage.get("text_encoder_root")
            if manual_root and repo in TEXT_REPOS[1:]:
                folder = absolute_folder(manual_root, "the text encoder root") / repo
            else:
                folder = Path(snapshot_download(repo_id=repo, cache_dir=hub, local_files_only=True))
            validate_text_snapshot(folder, repo)
            rows.append({"component": repo, "ready": True, "path": str(folder), "message": "Required local files present"})
        except Exception as exc:
            rows.append({"component": repo, "ready": False, "message": str(exc)})
    return {"ready": all(r["ready"] for r in rows), "components": rows,
            "note": "File availability check only; GPU memory, model deserialization and motion quality are not tested"}


def download_assets(request, status):
    """No inference, no prompt transmission. Called only for explicit download jobs."""
    if request.get("operation") not in DOWNLOAD_OPERATIONS or request.get("download_confirmed") is not True:
        raise ValueError("This operation does not authorize downloads")
    from huggingface_hub import snapshot_download
    storage = request["storage"]
    home = cache_home(storage)
    if home is None:
        raise ValueError("Choose a download/cache folder; refusing the system default")
    home.mkdir(parents=True, exist_ok=True)
    completed = []
    if request["operation"] == "download_model":
        repo = repository_id(request["model"])
        folder = checkpoint_folder(storage, request["model"], download=True)
        folder.mkdir(parents=True, exist_ok=True)
        state = folder / STATE_FILE
        if state.exists() and _json(state).get("repo_id") not in {None, repo}:
            raise ValueError("Destination belongs to another model; choose a different download folder")
        _write_json(state, {"state": "downloading", "repo_id": repo})
        status("downloading", f"Downloading/resuming {repo} to {folder}; see worker.log for transfer progress")
        snapshot_download(repo_id=repo, local_dir=str(folder), max_workers=4)
        checkpoint_config(folder, ignore_pending=True)
        _write_json(state, {"state": "complete", "repo_id": repo})
        completed.append(str(folder))
    if request["operation"] == "download_text" or request.get("include_text_encoder", False):
        for repo in TEXT_REPOS:
            status("downloading", f"Downloading/resuming local text encoder: {repo}")
            folder = Path(snapshot_download(repo_id=repo, cache_dir=str(home / "hub"),
                                           ignore_patterns=["original/*"], max_workers=4))
            validate_text_snapshot(folder, repo)
            completed.append(str(folder))
    return completed


def load_selected_model(request):
    """Instantiate a complete local config without upstream's online fallback.

Uses the same OmegaConf + Kimodo instantiate helper as upstream load_model.
No global monkey-patching, directory renaming, copying weights, or symlinks.
Older request payloads retain the upstream local-cache lookup, still offline.
"""
    if "storage" not in request:
        from kimodo import load_model
        return load_model(request.get("model", "Kimodo-SOMA-RP-v1.1"), device=request.get("device", "cuda:0"))
    from omegaconf import OmegaConf
    from kimodo.model.loading import instantiate_from_dict
    storage = request["storage"]
    folder = checkpoint_folder(storage, request["model"])
    conf, _ = checkpoint_config(folder)
    encoder = {"_target_": "kimodo.model.LLM2VecEncoder",
               "base_model_name_or_path": TEXT_REPOS[1], "peft_model_name_or_path": TEXT_REPOS[2],
               "dtype": "bfloat16", "llm_dim": 4096, "device": "auto"}
    runtime = OmegaConf.create({"checkpoint_dir": str(folder), "text_encoder": encoder})
    cfg = OmegaConf.to_container(OmegaConf.merge(conf, runtime), resolve=True)
    cfg.pop("checkpoint_dir", None)
    return instantiate_from_dict(cfg, overrides={"device": request.get("device", "cuda:0")}).eval()
