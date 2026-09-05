# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender UI and main-thread job lifecycle."""
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid
import bpy
from bpy.app.handlers import persistent
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty, FloatVectorProperty)
from bpy_extras.io_utils import ImportHelper, ExportHelper
from .timeline import load_json, parse_prompts, frame_prompts, export_prompts
from . import process, rig, storage_ui
from .model_storage import STORAGE_OPERATIONS, DOWNLOAD_OPERATIONS

_ACTIVE = None
_DRAW_HANDLE = None
_SUSPEND_UPDATES = False


def preferences(context):
    return context.preferences.addons[__package__].preferences


def enabled_prompts(scene):
    return parse_prompts([{"text": p.text, "duration": p.duration}
                          for p in scene.kimodo_studio.prompts if p.enabled])


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def timeline_origin(s):
    return s.source_end + 1 if s.preview_append else s.start_frame


def update_markers(scene):
    s = scene.kimodo_studio
    try:
        spans = frame_prompts(enabled_prompts(scene), rig.scene_fps(scene), timeline_origin(s))
    except ValueError:
        spans = []
    for marker in list(scene.timeline_markers):
        if marker.name.startswith("KMD::"):
            scene.timeline_markers.remove(marker)
    for i, span in enumerate(spans):
        scene.timeline_markers.new(f"KMD::{i+1:02d} {span.text}"[:63], frame=span.start)
    if spans:
        scene.timeline_markers.new("KMD::END", frame=spans[-1].end)
    redraw()


def edited(self, context):
    if not _SUSPEND_UPDATES and context and context.scene and hasattr(context.scene, "kimodo_studio"):
        update_markers(context.scene)


class KMD_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    backend: EnumProperty(name="Backend", items=[("NATIVE", "Native Python", "Windows/Linux Python environment"),
                                                   ("WSL", "WSL2", "Run Kimodo in a WSL2 environment")], default="NATIVE")
    python_path: StringProperty(name="Kimodo Python executable", subtype="FILE_PATH")
    repo_path: StringProperty(name="Kimodo repository", subtype="DIR_PATH")
    wsl_python: StringProperty(name="WSL venv Python", default="/home/USER/venvs/kimodo/bin/python")
    distro: StringProperty(name="WSL distribution", default="Ubuntu-22.04")
    output_root: StringProperty(name="Output/job folder", subtype="DIR_PATH")
    model_mode: EnumProperty(name="Model location", items=[
        ("MANAGED", "Download folder", "Select a preset and use its downloaded folder"),
        ("MANUAL", "Manual path", "Use an existing complete local checkpoint; do not modify it")])
    models_root: StringProperty(name="Model download folder", subtype="DIR_PATH")
    manual_model_path: StringProperty(name="Manual checkpoint folder", subtype="DIR_PATH")
    model_cache_root: StringProperty(name="Dependency cache folder", subtype="DIR_PATH")
    include_text_encoder: BoolProperty(name="Also download required text encoder", default=True)
    model_storage_details: BoolProperty(name="Dependency/cache options", default=False)
    text_encoder_device: StringProperty(name="Text encoder device", default="auto")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "backend")
        if self.backend == "WSL":
            layout.prop(self, "wsl_python"); layout.prop(self, "distro")
            layout.label(text="Repository/job files must be on Windows drives shared with WSL")
        else:
            layout.prop(self, "python_path")
        layout.prop(self, "repo_path"); layout.prop(self, "output_root")
        storage_ui.draw_storage(layout, context)
        layout.label(text="Use an existing Kimodo environment. Do not install torch inside Blender")
        layout.operator("kimodo.run_job", text="Check backend imports").operation = "check"


class KMD_Prompt(bpy.types.PropertyGroup):
    text: StringProperty(name="Motion prompt", default="A person walks with a strut", update=edited)
    duration: FloatProperty(name="Seconds", default=6., min=0.001, max=1200., precision=3, update=edited)
    enabled: BoolProperty(name="Enabled", default=True, update=edited)


class KMD_Constraint(bpy.types.PropertyGroup):
    kind: EnumProperty(name="Track", items=[(k, n, "") for k,n in (
        ("fullbody","Full Body"),("root2d","2D Root"),("left-hand","Left Hand"),
        ("right-hand","Right Hand"),("left-foot","Left Foot"),("right-foot","Right Foot"),
        ("end-effector","End Effectors"))])
    enabled: BoolProperty(name="Enabled", default=True, update=edited)
    start_frame: IntProperty(name="Start", default=1, update=edited)
    end_frame: IntProperty(name="End (inclusive)", default=1, update=edited)
    payload: StringProperty(default="{}")
    skeleton: StringProperty(default="{}")
    translation: FloatVectorProperty(name="Whole-pose offset", size=3, subtype="TRANSLATION", update=edited)
    target: PointerProperty(name="Waypoint target", type=bpy.types.Object)


class KMD_Take(bpy.types.PropertyGroup):
    rig: PointerProperty(type=bpy.types.Object)
    seed: IntProperty(default=-1)
    path: StringProperty(subtype="FILE_PATH")


class KMD_Settings(bpy.types.PropertyGroup):
    tab: EnumProperty(name="Section", items=[("GENERATE","Generate",""),("CONSTRAINTS","Constraints",""),
        ("FILES","Load / Save",""),("VISUALIZE","Visualize",""),("HELP","Help","")], default="GENERATE")
    model_report: StringProperty(default="")
    model_preset: EnumProperty(name="Model", items=storage_ui.PRESETS, get=storage_ui.preset_get, set=storage_ui.preset_set)
    dataset: EnumProperty(name="Training dataset", items=[("RP","Rigplay",""),("SEED","BONES-SEED","")])
    skeleton_choice: EnumProperty(name="Skeleton", items=[("SOMA","SOMA human",""),("G1","G1 robot",""),("SMPLX","SMPL-X human","")])
    version_choice: EnumProperty(name="Version", items=[("v1.1","v1.1","SOMA only"),("v1","v1","")])
    custom_model: BoolProperty(name="Custom model identifier", default=False)
    heading: FloatProperty(name="Initial heading", subtype="ANGLE", default=0.)
    align_first: BoolProperty(name="Align generation to the first constraint", default=True)
    root_margin: FloatProperty(name="Root cleanup margin (m)", min=0., max=2., default=.04)
    num_samples: IntProperty(name="Variations", default=1, min=1, max=8)
    rig_source: PointerProperty(name="Source rig", type=bpy.types.Object, poll=lambda self,obj: obj.type=="ARMATURE")
    constraints: CollectionProperty(type=KMD_Constraint)
    constraint_index: IntProperty(default=0)
    constraint_kind: EnumProperty(name="Track", items=[(k,n,"") for k,n in (
        ("fullbody","Full Body"),("root2d","2D Root"),("left-hand","Left Hand"),
        ("right-hand","Right Hand"),("left-foot","Left Foot"),("right-foot","Right Foot"))])
    capture_start: IntProperty(name="Interval start", default=1)
    capture_end: IntProperty(name="Interval end", default=30)
    constraint_import_fps: FloatProperty(name="Native constraints JSON FPS", default=30., min=1., max=240.)
    cursor_waypoint: BoolProperty(name="Root waypoint from 3D cursor", default=False)
    edit_object: PointerProperty(type=bpy.types.Object)
    editing_index: IntProperty(default=-1)
    takes: CollectionProperty(type=KMD_Take)
    take_index: IntProperty(default=0)
    show_tracks: BoolProperty(name="Show constraint tracks", default=True)
    row_height: IntProperty(name="Track height", default=26, min=20, max=44)
    display_type: EnumProperty(name="Skeleton display", items=[("STICK","Stick",""),("OCTAHEDRAL","Bones",""),("WIRE","Wire","")])
    show_names: BoolProperty(name="Bone names", default=False)
    in_front: BoolProperty(name="Skeleton in front", default=True)
    show_floor: BoolProperty(name="Ground grid", default=True)
    viewport_shading: EnumProperty(name="Viewport shading", items=[("SOLID","Solid",""),("MATERIAL","Material preview","")])
    prompts: CollectionProperty(type=KMD_Prompt)
    index: IntProperty(default=0)
    start_frame: IntProperty(name="Generate at frame", default=1, update=edited)
    source_start: IntProperty(name="Source start", default=1)
    source_end: IntProperty(name="Source end", default=180, update=edited)
    preview_append: BoolProperty(name="Show prompts after source end", default=False, update=edited)
    overlay: BoolProperty(name="Show prompt bars in Dope Sheet", default=True, update=edited)
    model: StringProperty(name="Model", default="Kimodo-SOMA-RP-v1.1")
    device: StringProperty(name="Torch device", default="cuda:0")
    steps: IntProperty(name="Denoising steps", default=50, min=1, max=1000)
    seed: IntProperty(name="Seed (-1 = random)", default=42, min=-1)
    transition: IntProperty(name="Between-prompt transition frames", default=5, min=1, max=120)
    context: IntProperty(name="Source context frames (model FPS)", default=12, min=2, max=120)
    blend: IntProperty(name="Blend tail frames (0 preserves source)", default=0, min=0, max=120)
    source_fps: FloatProperty(name="Import FPS override (0 = auto)", default=0., min=0., max=240.)
    text_guidance: FloatProperty(name="Text guidance", default=2., min=0., max=20.)
    constraint_guidance: FloatProperty(name="Constraint guidance", default=2., min=0., max=20.)
    postprocess: BoolProperty(name="Kimodo foot-skate/constraint cleanup", default=False)
    export_bvh: BoolProperty(name="Also export BVH for SOMA", default=True)
    status: StringProperty(default="Ready — configure the external backend in add-on preferences")
    last_job: StringProperty(name="Last job", subtype="DIR_PATH")
    last_motion: StringProperty(name="Last motion", subtype="FILE_PATH")


class KMD_UL_Prompts(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.label(text=str(index+1))
        row.prop(item, "text", text="", emboss=False)
        row.prop(item, "duration", text="s")


class KMD_OT_Edit(bpy.types.Operator):
    bl_idname = "kimodo.edit_prompt"
    bl_label = "Edit prompt list"
    bl_options = {"REGISTER", "UNDO"}
    action: EnumProperty(items=[(x, x.title(), "") for x in ("ADD", "REMOVE", "UP", "DOWN", "REFRESH")])

    def execute(self, context):
        s = context.scene.kimodo_studio
        if self.action == "REFRESH":
            pass
        elif self.action == "ADD":
            s.prompts.add(); s.index = len(s.prompts)-1
        elif s.prompts:
            i = min(s.index, len(s.prompts)-1)
            if self.action == "REMOVE":
                s.prompts.remove(i); s.index = max(0, i-1)
            else:
                target = max(0, min(len(s.prompts)-1, i + (-1 if self.action == "UP" else 1)))
                s.prompts.move(i, target); s.index = target
        update_markers(context.scene)
        return {"FINISHED"}


class KMD_OT_ImportJSON(bpy.types.Operator, ImportHelper):
    bl_idname = "kimodo.import_prompts"
    bl_label = "Import prompt JSON"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})
    append: BoolProperty(name="Append to existing prompts", default=False)

    def execute(self, context):
        try:
            prompts = parse_prompts(load_json(self.filepath))
            frame_prompts(prompts, rig.scene_fps(context.scene))
            s = context.scene.kimodo_studio
            # Validate the FULL candidate before touching scene data.
            existing = ([{"text": p.text, "duration": p.duration} for p in s.prompts] if self.append else [])
            combined = parse_prompts(existing + [{"text": p.text, "duration": float(p.duration)} for p in prompts])
            # Preserve existing mute flags when appending. Only new entries are created.
            if not self.append:
                s.prompts.clear()
            for p in (prompts if self.append else combined):
                item = s.prompts.add(); item.text = p.text; item.duration = float(p.duration)
            s.index = 0
            update_markers(context.scene)
        except (ValueError, OSError, KeyError) as exc:
            self.report({"ERROR"}, str(exc)); return {"CANCELLED"}
        self.report({"INFO"}, f"Imported {len(prompts)} timed prompts")
        return {"FINISHED"}


class KMD_OT_ExportJSON(bpy.types.Operator, ExportHelper):
    bl_idname = "kimodo.export_prompts"
    bl_label = "Export prompt JSON"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})
    native: BoolProperty(name="Kimodo meta.json format", default=False)

    def execute(self, context):
        try:
            data = export_prompts(enabled_prompts(context.scene), native=self.native)
            Path(self.filepath).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (ValueError, OSError) as exc:
            self.report({"ERROR"}, str(exc)); return {"CANCELLED"}
        return {"FINISHED"}


class KMD_OT_ImportMotion(bpy.types.Operator, ImportHelper):
    bl_idname = "kimodo.import_motion"
    bl_label = "Import native Kimodo animation"
    filter_glob: StringProperty(default="*.bvh;*.npz;*.csv", options={"HIDDEN"})

    def execute(self, context):
        return bpy.ops.kimodo.run_job("INVOKE_DEFAULT", operation="import", source_file=self.filepath)


class KMD_OT_SourceRange(bpy.types.Operator):
    bl_idname = "kimodo.source_range"
    bl_label = "Use active Action range"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.scene.kimodo_studio.rig_source or context.object
        if not obj or not obj.animation_data or not obj.animation_data.action:
            self.report({"ERROR"}, "Select an animated armature with an active Action")
            return {"CANCELLED"}
        a, b = obj.animation_data.action.frame_range
        s = context.scene.kimodo_studio
        s.rig_source = obj
        s.source_start, s.source_end = math.ceil(a), math.floor(b)
        s.preview_append = True
        return {"FINISHED"}


def stop_process(job):
    proc = job.get("process")
    if proc and proc.poll() is None:
        if job.get("backend") == "WSL":
            # Stop only this worker, never terminate an entire WSL distribution.
            try:
                pid_info = json.loads((job["folder"] / "worker_pid.json").read_text())
                pid = int(pid_info["pid"])
                if pid > 1:
                    subprocess.run(["wsl.exe", "--distribution", job["distro"], "--exec",
                                    "kill", "-TERM", str(pid)], timeout=5, check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


def cleanup_job(job, *, cancel=False):
    global _ACTIVE
    if cancel:
        stop_process(job)
        baker = job.get("baker")
        if baker:
            baker.close()
    if job.get("timer"):
        try:
            job["wm"].event_timer_remove(job["timer"])
        except (ReferenceError, RuntimeError):
            pass
    if job.get("log"):
        job["log"].close()
    _ACTIVE = None


@persistent
def file_load_pre(_):
    if _ACTIVE:
        cleanup_job(_ACTIVE, cancel=True)


class KMD_OT_Cancel(bpy.types.Operator):
    bl_idname = "kimodo.cancel_job"
    bl_label = "Cancel Kimodo job"
    def execute(self, context):
        if _ACTIVE:
            _ACTIVE["cancel"] = True
        return {"FINISHED"}


class KMD_OT_Run(bpy.types.Operator):
    bl_idname = "kimodo.run_job"
    bl_label = "Run Kimodo"
    operation: EnumProperty(items=[(x, x.title(), "") for x in ("check", "import", "generate", "continue", "export", "prepare", "check_models", "download_model", "download_text")])
    download_confirmed: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})
    source_file: StringProperty(subtype="FILE_PATH")

    @classmethod
    def poll(cls, context):
        return (_ACTIVE is None and context.mode == "OBJECT" and context.scene is not None
                and context.scene.kimodo_studio.editing_index < 0)

    def execute(self, context):
        return self.invoke(context, None)

    def invoke(self, context, event):
        global _ACTIVE
        if _ACTIVE:
            self.report({"ERROR"}, "A Kimodo job is already running"); return {"CANCELLED"}
        scene = context.scene
        s, pref = scene.kimodo_studio, preferences(context)
        logfile = None
        try:
            output = Path(bpy.path.abspath(pref.output_root)).expanduser() if pref.output_root else Path.home()/"KimodoOutputs"
            folder = output / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
            python = pref.wsl_python if pref.backend == "WSL" else bpy.path.abspath(pref.python_path)
            repo = bpy.path.abspath(pref.repo_path)
            worker = Path(__file__).with_name("worker.py")
            command = process.build_command(python, repo, worker, folder, pref.backend, pref.distro)
            from . import studio
            from .authoring import model_name
            selected_model = (s.model if s.custom_model or self.operation in {"check", "import", "export"}
                              else model_name(s.dataset, s.skeleton_choice, s.version_choice))
            request = {"schema_version": 1, "operation": self.operation,
                       "model": selected_model, "device": s.device, "output_fps": rig.scene_fps(scene),
                       "seed": s.seed, "steps": s.steps, "transition_frames": s.transition,
                       "context_frames": s.context, "blend_frames": s.blend,
                       "text_guidance": s.text_guidance, "constraint_guidance": s.constraint_guidance,
                       "postprocess": s.postprocess, "export_bvh": s.export_bvh,
                       "storage": storage_ui.storage_request(pref),
                       "download_confirmed": self.download_confirmed,
                       "include_text_encoder": pref.include_text_encoder,
                       "num_samples": s.num_samples, "root_margin": s.root_margin, "heading": s.heading, "align_first": s.align_first,
                       "scene_fps": rig.scene_fps(scene), "sequence_origin": s.source_end+1 if self.operation=="continue" else s.start_frame}
            if self.operation in DOWNLOAD_OPERATIONS and not self.download_confirmed:
                raise ValueError("Use the Download / resume button and confirm its destination")
            if self.operation in {"generate", "continue", "prepare", "download_model"}:
                from .model_storage import checkpoint_folder
                checkpoint_folder(storage_ui.storage_request(pref, translate=False), selected_model,
                                  download=self.operation == "download_model")
            if self.operation in {"generate", "continue"}:
                request["timeline"] = export_prompts(enabled_prompts(scene))
                spans = frame_prompts(enabled_prompts(scene), rig.scene_fps(scene))
                request["constraints"] = studio.constraint_records(s)
                from .authoring import compile_constraints
                compile_constraints(request["constraints"], rig.scene_fps(scene), rig.scene_fps(scene),
                                    request["sequence_origin"], spans[-1].end)
            if self.operation == "continue" and (s.blend == 1 or s.blend > s.context):
                raise ValueError("Blend frames must be 0 or between 2 and context frames")
            folder.mkdir(parents=True, exist_ok=False)
            start = s.start_frame
            if self.operation in {"continue", "export"}:
                prefix = folder / "source.blender.npz"
                rig.snapshot(s.rig_source or context.object, scene, s.source_start, s.source_end, prefix)
                source = str(prefix); start = s.source_start
            else:
                source = self.source_file
            if source:
                request["source"] = process.wsl_path(source) if pref.backend == "WSL" else source
                if self.operation == "import" and s.source_fps > 0:
                    request["source_fps"] = s.source_fps
            (folder / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
            logfile = (folder / "worker.log").open("w", encoding="utf-8")
            proc = process.launch(command, logfile)
            wm = context.window_manager
            timer = wm.event_timer_add(0.1, window=context.window)
            _ACTIVE = {"process": proc, "log": logfile, "timer": timer, "wm": wm,
                       "folder": folder, "scene": scene, "start": start, "operation": self.operation,
                       "backend": pref.backend, "distro": pref.distro, "baker": None, "cancel": False,
                       "queue": [], "finished": 0, "samples": [],
                       "storage_key": storage_ui.fingerprint(selected_model, request["storage"])}
            self._job = _ACTIVE
            s.last_job = str(folder); s.status = "Kimodo worker started"
            wm.modal_handler_add(self)
            return {"RUNNING_MODAL"}
        except Exception as exc:
            if logfile:
                logfile.close()
            self.report({"ERROR"}, str(exc)); s.status = str(exc)
            return {"CANCELLED"}

    def modal(self, context, event):
        global _ACTIVE
        job = self._job
        if _ACTIVE is not job:
            return {"CANCELLED"}
        if event.type == "ESC" or job["cancel"]:
            job["scene"].kimodo_studio.status = (
                "Cancelled; completed files retained, use Download / resume to continue"
                if job["operation"] in DOWNLOAD_OPERATIONS else "Cancelled; original animation retained")
            cleanup_job(job, cancel=True); redraw(); return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        scene, folder = job["scene"], job["folder"]
        if context.scene != scene:
            scene.kimodo_studio.status = "Cancelled because active scene changed"
            cleanup_job(job, cancel=True); return {"CANCELLED"}
        s = scene.kimodo_studio
        try:
            if job["baker"] is not None:
                deadline = time.monotonic() + 0.025
                while time.monotonic() < deadline:
                    try:
                        frame, total, obj = next(job["baker"])
                        job["result_object"] = obj
                        s.status = f"Baking new Action: {frame}/{total} frames"
                    except StopIteration:
                        obj = job.get("result_object")
                        if obj:
                            take = s.takes.add(); take.name = obj.name; take.rig = obj
                            take.seed = int(job.get("sample_seed", -1))
                            take.path = s.last_motion; s.take_index = len(s.takes)-1
                            if job["operation"] in {"import", "prepare"}:
                                s.rig_source = obj
                                s.source_start = math.ceil(obj["kimodo_frame_start"])
                                s.source_end = math.floor(obj["kimodo_frame_end"])
                        job["finished"] += 1; job["baker"] = None
                        if job["queue"]:
                            self.start_bake(job)
                            return {"RUNNING_MODAL"}
                        s.status = f"Done — {job['finished']} new take(s); source Action unchanged"
                        cleanup_job(job); redraw(); return {"FINISHED"}
                redraw(); return {"RUNNING_MODAL"}
            status_path = folder / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            s.status = status.get("message", "Starting external Python environment")
            code = job["process"].poll()
            if code is None:
                redraw(); return {"RUNNING_MODAL"}
            if code != 0 or status.get("state") != "done":
                raise RuntimeError(status.get("message", f"Worker exited with {code}. Open worker.log"))
            if job["operation"] in STORAGE_OPERATIONS:
                report = status.get("model_report", {})
                report["key"] = job["storage_key"]
                s.model_report = json.dumps(report)
                s.status = status.get("message", "Model storage job complete")
                cleanup_job(job); redraw(); return {"FINISHED"}
            if job["operation"] in {"check", "export"}:
                s.status = status.get("message", "Done") + f" — {folder}"
                cleanup_job(job); redraw(); return {"FINISHED"}
            samples = status.get("samples", [{"directory": ".", "seed": s.seed}])
            for sample in samples:
                relative = Path(sample["directory"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe sample path in worker response")
                job["queue"].append((folder / relative / "motion.blender.npz", sample.get("seed", -1)))
            self.start_bake(job)
            return {"RUNNING_MODAL"}
        except Exception as exc:
            s.status = str(exc)
            self.report({"ERROR"}, str(exc))
            cleanup_job(job, cancel=True); redraw(); return {"CANCELLED"}


    def start_bake(self, job):
        path, seed = job["queue"].pop(0)
        job["sample_seed"] = seed
        job["scene"].kimodo_studio.last_motion = str(path)
        label = "Kimodo Continued" if job["operation"]=="continue" else "Kimodo Motion"
        job["baker"] = rig.bake_motion(path, job["scene"], job["start"], label)


class KMD_OT_OpenOutput(bpy.types.Operator):
    bl_idname = "kimodo.open_output"
    bl_label = "Open last job/output folder"
    def execute(self, context):
        path = context.scene.kimodo_studio.last_job
        if path and Path(path).is_dir():
            bpy.ops.wm.path_open(filepath=path)
            return {"FINISHED"}
        return {"CANCELLED"}


def draw_panel(layout, context):
    from . import studio
    studio.draw_panel(layout, context)


class KMD_PT_View(bpy.types.Panel):
    bl_label = "Kimodo Motion Studio"
    bl_idname = "KMD_PT_view"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Kimodo"
    def draw(self, context):
        draw_panel(self.layout, context)


class KMD_PT_Dopesheet(bpy.types.Panel):
    bl_label = "Kimodo Motion Studio"
    bl_idname = "KMD_PT_dopesheet"
    bl_space_type = "DOPESHEET_EDITOR"
    bl_region_type = "UI"
    bl_category = "Kimodo"
    def draw(self, context):
        draw_panel(self.layout, context)


def draw_prompt_overlay():
    from . import interaction
    interaction.draw_timeline()


CLASSES = (KMD_Preferences, KMD_Prompt, KMD_Constraint, KMD_Take, KMD_Settings, KMD_UL_Prompts, KMD_OT_Edit,
           KMD_OT_ImportJSON, KMD_OT_ExportJSON, KMD_OT_ImportMotion, KMD_OT_SourceRange,
           KMD_OT_Cancel, KMD_OT_Run, KMD_OT_OpenOutput, KMD_PT_View, KMD_PT_Dopesheet)


def register():
    global _DRAW_HANDLE
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.kimodo_studio = PointerProperty(type=KMD_Settings)
    _DRAW_HANDLE = bpy.types.SpaceDopeSheetEditor.draw_handler_add(draw_prompt_overlay, (), "WINDOW", "POST_PIXEL")
    bpy.app.handlers.load_pre.append(file_load_pre)
    from . import studio, interaction
    studio.register()
    storage_ui.register()
    interaction.register()
    studio.migrate_legacy()


def unregister():
    global _DRAW_HANDLE
    if _ACTIVE:
        cleanup_job(_ACTIVE, cancel=True)
    if file_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(file_load_pre)
    if _DRAW_HANDLE is not None:
        bpy.types.SpaceDopeSheetEditor.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
        _DRAW_HANDLE = None
    from . import studio, interaction
    interaction.unregister()
    storage_ui.unregister()
    studio.unregister()
    if hasattr(bpy.types.Scene, "kimodo_studio"):
        del bpy.types.Scene.kimodo_studio
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
