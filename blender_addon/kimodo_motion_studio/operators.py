import json
from pathlib import Path

import bpy
from bpy_extras.io_utils import ImportHelper


def _normalize_prompt_payload(payload):
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and "prompts" in payload:
        items = payload["prompts"]
    elif isinstance(payload, dict) and "texts" in payload and "durations" in payload:
        texts = payload["texts"]
        durations = payload["durations"]
        if len(texts) != len(durations):
            raise ValueError("texts/durations length mismatch")
        items = [{"text": t, "duration": d} for t, d in zip(texts, durations)]
    elif isinstance(payload, dict) and "text" in payload and "duration" in payload:
        items = [payload]
    else:
        raise ValueError("Unsupported prompt JSON format")

    normalized = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt entry {i} must be an object")
        text = str(item.get("text", "")).strip()
        duration = float(item.get("duration", 0.0))
        if not text:
            raise ValueError(f"Prompt entry {i} has no text")
        if duration <= 0:
            raise ValueError(f"Prompt entry {i} duration must be > 0")
        normalized.append((text, duration))
    return normalized


class KIMODO_OT_import_prompts_json(bpy.types.Operator, ImportHelper):
    bl_idname = "kimodo.import_prompts_json"
    bl_label = "Import Prompt JSON"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        settings = context.scene.kimodo_motion
        try:
            payload = json.loads(Path(self.filepath).read_text(encoding="utf-8"))
            prompts = _normalize_prompt_payload(payload)
        except Exception as exc:
            self.report({'ERROR'}, f"Prompt import failed: {exc}")
            return {'CANCELLED'}

        settings.prompts.clear()
        for text, duration in prompts:
            item = settings.prompts.add()
            item.text = text
            item.duration = duration
            item.enabled = True
        settings.prompt_index = 0
        self.report({'INFO'}, f"Imported {len(prompts)} prompt(s)")
        return {'FINISHED'}


class KIMODO_OT_add_prompt(bpy.types.Operator):
    bl_idname = "kimodo.add_prompt"
    bl_label = "Add Prompt"

    def execute(self, context):
        item = context.scene.kimodo_motion.prompts.add()
        item.text = "A person walks with a strut"
        item.duration = 6.0
        context.scene.kimodo_motion.prompt_index = len(context.scene.kimodo_motion.prompts) - 1
        return {'FINISHED'}


class KIMODO_OT_remove_prompt(bpy.types.Operator):
    bl_idname = "kimodo.remove_prompt"
    bl_label = "Remove Prompt"

    def execute(self, context):
        settings = context.scene.kimodo_motion
        if settings.prompts:
            idx = max(0, min(settings.prompt_index, len(settings.prompts) - 1))
            settings.prompts.remove(idx)
            settings.prompt_index = max(0, min(idx, len(settings.prompts) - 1))
        return {'FINISHED'}


class KIMODO_OT_build_markers(bpy.types.Operator):
    bl_idname = "kimodo.build_prompt_markers"
    bl_label = "Build Timeline Markers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        settings = scene.kimodo_motion
        fps = scene.render.fps / scene.render.fps_base

        for marker in list(scene.timeline_markers):
            if marker.name.startswith("KIMODO:"):
                scene.timeline_markers.remove(marker)

        frame = scene.frame_start
        for index, prompt in enumerate(settings.prompts):
            if not prompt.enabled:
                continue
            scene.timeline_markers.new(f"KIMODO:{index + 1} {prompt.text[:48]}", frame=int(round(frame)))
            frame += max(1, int(round(prompt.duration * fps)))

        scene.frame_end = max(scene.frame_end, int(round(frame - 1)))
        return {'FINISHED'}


_CLASSES = (
    KIMODO_OT_import_prompts_json,
    KIMODO_OT_add_prompt,
    KIMODO_OT_remove_prompt,
    KIMODO_OT_build_markers,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
