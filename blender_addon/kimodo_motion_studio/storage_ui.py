# SPDX-License-Identifier: GPL-3.0-or-later
"""Native Blender model-location and explicit-download controls."""
import json
from pathlib import Path
import bpy
from bpy.props import EnumProperty
from .model_storage import MODEL_NAMES, checkpoint_folder, cache_home
from . import process

PRESETS = [(name, name, "Local NVIDIA Kimodo preset", index) for index, name in enumerate(MODEL_NAMES)]
PRESETS.append(("CUSTOM", "Custom model identifier", "Use an explicit owner/model ID", len(MODEL_NAMES)))


def selected_model(settings):
    from .authoring import model_name
    return settings.model if settings.custom_model else model_name(
        settings.dataset, settings.skeleton_choice, settings.version_choice)


def preset_get(settings):
    try:
        name = selected_model(settings)
        return MODEL_NAMES.index(name)
    except ValueError:
        return len(MODEL_NAMES)


def preset_set(settings, value):
    settings.custom_model = value == len(MODEL_NAMES)
    if not settings.custom_model:
        name = MODEL_NAMES[value]
        _, skeleton, dataset, version = name.split("-")
        settings.skeleton_choice, settings.dataset, settings.version_choice = skeleton, dataset, version
        settings.model = name


def storage_request(pref, *, translate=True):
    def path(value):
        if not value.strip(): return ""
        value = str(Path(bpy.path.abspath(value)).expanduser())
        return process.wsl_path(value) if translate and pref.backend == "WSL" else value
    return {"mode": pref.model_mode, "models_root": path(pref.models_root),
            "manual_path": path(pref.manual_model_path), "cache_root": path(pref.model_cache_root),
            "text_encoder_device": pref.text_encoder_device}


def fingerprint(model, storage):
    return json.dumps({"model": model, "storage": storage}, sort_keys=True)


def draw_storage(layout, context):
    from . import ui
    pref, settings = ui.preferences(context), context.scene.kimodo_studio
    box = layout.box()
    box.label(text="Local model storage", icon="DISK_DRIVE")
    box.prop(pref, "model_mode", expand=True)
    if pref.model_mode == "MANUAL":
        box.prop(pref, "manual_model_path")
        box.label(text="Complete folder: config.yaml, weights and stats")
    box.prop(pref, "models_root")
    try:
        model = selected_model(settings)
        data = storage_request(pref, translate=False)
        target = checkpoint_folder(data, model)
        row = box.row(); row.scale_y = 0.8
        row.label(text="Checkpoint: " + str(target))
    except ValueError as exc:
        box.label(text=str(exc), icon="INFO")
    row = box.row(align=True)
    row.operator("kimodo.run_job", text="Check local files", icon="VIEWZOOM").operation = "check_models"
    download = row.row(); download.enabled = pref.model_mode == "MANAGED"
    download.operator("kimodo.download_assets", text="Download / resume", icon="IMPORT").kind = "download_model"
    if pref.model_mode == "MANAGED":
        box.prop(pref, "include_text_encoder")
    box.prop(pref, "model_storage_details", text="Dependency/cache options", icon="PREFERENCES")
    if pref.model_storage_details:
        box.prop(pref, "model_cache_root")
        box.label(text="Blank cache: Download folder / .huggingface")
        box.prop(pref, "text_encoder_device")
        box.operator("kimodo.download_assets", text="Download text encoder only").kind = "download_text"
        box.label(text="Existing HF login is reused; gated models need access")
    box.label(text="Generation is local/offline; downloads are separate", icon="LOCKED")
    try:
        report = json.loads(settings.model_report or "{}")
        key = fingerprint(selected_model(settings), storage_request(pref))
        if report.get("key") == key:
            for item in report.get("components", []):
                name = item["component"].rsplit("/", 1)[-1]
                box.label(text=("Ready: " if item["ready"] else "Missing: ") + name,
                          icon="CHECKMARK" if item["ready"] else "ERROR")
            box.label(text="File check only; not an inference test")
        else:
            box.label(text="Run Check local files after selecting a model/path")
    except (ValueError, TypeError):
        pass


class KMD_OT_Download(bpy.types.Operator):
    bl_idname = "kimodo.download_assets"
    bl_label = "Confirm local model download"
    kind: EnumProperty(items=[("download_model", "Selected model", ""),
                              ("download_text", "Text encoder", "")])

    @classmethod
    def poll(cls, context):
        from . import ui
        return (ui._ACTIVE is None and context.scene is not None and context.mode == "OBJECT"
                and context.scene.kimodo_studio.editing_index < 0)

    def invoke(self, context, event):
        from . import ui
        try:
            pref = ui.preferences(context)
            storage = storage_request(pref, translate=False)
            if self.kind == "download_model":
                checkpoint_folder(storage, selected_model(context.scene.kimodo_studio), download=True)
            if cache_home(storage) is None:
                raise ValueError("Choose the model download folder or dependency cache folder first")
        except ValueError as exc:
            self.report({"ERROR"}, str(exc)); return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, width=580)

    def draw(self, context):
        from . import ui
        pref = ui.preferences(context)
        storage = storage_request(pref, translate=False)
        self.layout.label(text="Downloads use Hugging Face; generation stays on this computer.")
        if self.kind == "download_model":
            model = selected_model(context.scene.kimodo_studio)
            self.layout.label(text="Model: " + model)
            self.layout.label(text="To: " + str(checkpoint_folder(storage, model, download=True)))
        if self.kind == "download_text" or pref.include_text_encoder:
            self.layout.label(text="Includes Llama-3 base weights and two LLM2Vec adapters.")
            self.layout.label(text="Text weights can require many GB and Meta/HF account approval.")
        self.layout.label(text="Cache: " + str(cache_home(storage)))
        self.layout.label(text="Existing downloads are reused. No prompts or animations are uploaded.")
        self.layout.label(text="Model licenses apply; this does not accept gated licenses for you.")

    def execute(self, context):
        # Programmatic execution is also an explicit Download action.
        return bpy.ops.kimodo.run_job("INVOKE_DEFAULT", operation=self.kind, download_confirmed=True)


def register():
    bpy.utils.register_class(KMD_OT_Download)


def unregister():
    bpy.utils.unregister_class(KMD_OT_Download)
