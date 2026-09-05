# SPDX-License-Identifier: GPL-3.0-or-later
"""External Kimodo adapter. Uses the checked upstream public API; no bpy imports."""
import json
import math
import os
from pathlib import Path
import re
import numpy as np
from .model_storage import (configure_environment, STORAGE_OPERATIONS, DOWNLOAD_OPERATIONS,
                            inspect_storage, download_assets, load_selected_model)
from .timeline import load_json, parse_prompts, frame_prompts, round_frame
from .motion_math import Clip, resample, join_context, safe_npz, load_portable, save_portable


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def _numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _single(x, rank):
    x = _numpy(x)
    if x.ndim == rank + 1 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != rank:
        raise ValueError(f"Unexpected Kimodo output shape: {x.shape}")
    return x


class KimodoAdapter:
    def __init__(self):
        import torch
        from kimodo.skeleton.registry import build_skeleton
        from kimodo.exports.motion_io import complete_motion_dict
        self.torch, self.build, self.complete = torch, build_skeleton, complete_motion_dict

    def tensor(self, x, skeleton):
        return self.torch.as_tensor(np.array(x, copy=True), dtype=self.torch.float32,
                                    device=skeleton.neutral_joints.device)

    def derive(self, clip, skeleton):
        return self.complete(self.tensor(clip.local, skeleton), self.tensor(clip.root, skeleton), skeleton, clip.fps)

    def metadata(self, skeleton):
        # The FK neutral pose, not a guessed bone layout or a guessed joint count.
        t = self.torch
        device = skeleton.neutral_joints.device
        ident = t.eye(3, device=device).repeat(1, skeleton.nbjoints, 1, 1)
        rot, pos, _ = skeleton.fk(ident, t.zeros((1, 3), device=device))
        return {"name": skeleton.name, "names": list(skeleton.bone_order_names),
                "parents": _numpy(skeleton.joint_parents).astype(int).tolist(),
                "root_idx": int(skeleton.root_idx), "rest_joints": _numpy(pos[0]).tolist(),
                "rest_global_rot_mats": _numpy(rot[0]).tolist(), "up_axis": "Y", "unit": "meter"}

    def source(self, path, source_fps=None):
        from kimodo.exports.motion_io import load_motion_file
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".npz":
            raw = safe_npz(path)
            if "kimodo_blender_schema" in raw:
                clip, meta, _ = load_portable(path)
                sk = self.build(clip.local.shape[1])
                expected = self.metadata(sk)
                if meta["names"] != expected["names"] or meta["parents"] != expected["parents"]:
                    raise ValueError("Source rig does not match Kimodo joint names and hierarchy")
                if not np.allclose(meta["rest_joints"], expected["rest_joints"], atol=0.001):
                    raise ValueError("Source rig proportions differ from the model. Retarget first")
                if not np.allclose(meta["rest_global_rot_mats"], expected["rest_global_rot_mats"], atol=0.001):
                    raise ValueError("Source rest rotations differ from the model. Retarget first")
                return clip, sk
            auto_fps = float(raw.get("mocap_frame_rate", raw.get("fps", 30.0)))
        elif path.suffix.lower() == ".bvh":
            auto_fps = None
            with path.open(encoding="utf-8-sig") as f:
                for line in f:
                    if line.strip().lower().startswith("frame time:"):
                        auto_fps = 1. / float(line.split(":", 1)[1]); break
            if auto_fps is None:
                raise ValueError("BVH is missing its Frame Time header")
        elif path.suffix.lower() == ".csv":
            auto_fps = 30.0
        else:
            raise ValueError("Backend import supports SOMA BVH, Kimodo/AMASS NPZ and G1 CSV only")
        rate = float(source_fps or auto_fps)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Invalid source frame rate")
        # Import at native FPS. Our resampler also handles identical quaternions correctly.
        d, j = load_motion_file(str(path), source_fps=rate, target_fps=rate)
        clip = Clip(_single(d["local_rot_mats"], 4), _single(d["root_positions"], 2), rate).validate()
        return clip, self.build(j)

    def save(self, folder, clip, skeleton, *, export_bvh=True):
        from kimodo.exports.motion_io import save_kimodo_npz
        d = self.derive(clip, skeleton)
        save_kimodo_npz(str(folder / "motion.npz"), d)
        # Native Kimodo NPZ historically lacks FPS; include a sidecar and Blender transfer file.
        atomic_json(folder / "motion_info.json", {"fps": clip.fps, "skeleton": skeleton.name})
        save_portable(folder / "motion.blender.npz", clip, self.metadata(skeleton),
                      {k: _numpy(v) for k, v in d.items()})
        files = ["motion.npz", "motion.blender.npz", "motion_info.json"]
        if export_bvh and "soma" in skeleton.name.lower():
            from kimodo.exports.bvh import save_motion_bvh
            save_motion_bvh(str(folder / "motion.bvh"), self.tensor(clip.local, skeleton),
                            self.tensor(clip.root, skeleton), skeleton=skeleton, fps=clip.fps,
                            standard_tpose=False)
            files.append("motion.bvh")
        return files

    def generate(self, request, status):
        from kimodo.constraints import FullBodyConstraintSet, EndEffectorConstraintSet
        from kimodo.motion_rep.feature_utils import compute_heading_angle
        from kimodo.tools import seed_everything
        prompts = parse_prompts(request["timeline"])
        status("loading_model", "Loading Kimodo model and text encoder")
        key = (request.get("model", "Kimodo-SOMA-RP-v1.1"), request.get("device", "cuda:0"),
               json.dumps(request.get("storage", {}), sort_keys=True))
        if getattr(self, "_model_key", None) != key:
            self._model = load_selected_model(request)
            self._model_key = key
        model = self._model
        fps = float(model.fps)
        spans = frame_prompts(prompts, fps)
        counts = [p.count for p in spans]
        if min(counts) < 2:
            raise ValueError("Every generated prompt needs at least two model frames")
        transition = int(request.get("transition_frames", 5))
        if len(counts) > 1 and not 1 <= transition < min(counts):
            raise ValueError("Transition frames must be positive and shorter than every prompt")
        steps = int(request.get("steps", 50))
        if not 1 <= steps <= 1000:
            raise ValueError("Denoising steps must be between 1 and 1000")
        seed = int(request.get("seed", 42))
        if seed >= 0:
            seed_everything(seed)
        k, source, source_sk = 0, None, None
        constraints, first_heading = [], float(request.get("heading", 0.0))
        origin = np.zeros(3, dtype=np.float32)
        train = model.skeleton
        out_skeleton = self.build(77 if train.nbjoints == 30 else train.nbjoints)
        if request["operation"] == "continue":
            source, source_sk = self.source(request["source"], request.get("source_fps"))
            if source.local.shape[1] != out_skeleton.nbjoints:
                raise ValueError("Source skeleton differs from model output skeleton. Retarget before continuation")
            context_source = resample(source, fps)
            k = int(request.get("context_frames", 12))
            if not 2 <= k <= len(context_source.root):
                raise ValueError("Context needs 2 or more real source frames, within the source clip length")
            tail = Clip(context_source.local[-k:].copy(), context_source.root[-k:].copy(), fps)
            # Horizontal translation only. Retain world yaw and pass the first heading explicitly.
            origin = tail.root[0].copy(); origin[1] = 0.
            tail.root -= origin
            train_local = self.tensor(tail.local, train)
            if train.nbjoints == 30 and tail.local.shape[1] == 77:
                train_local = train.from_SOMASkeleton77(train_local)
            d = self.complete(train_local, self.tensor(tail.root, train), train, fps)
            # Use CPU constraint data, as in upstream saved-example constraints. The model
            # condition builder moves the resulting feature tensors to its target device.
            constraint_sk = self.build(train.nbjoints)
            pos = d["posed_joints"].detach().cpu()
            rot = d["global_rot_mats"].detach().cpu()
            frames = self.torch.arange(k)
            constraints = [FullBodyConstraintSet(constraint_sk, frames, pos, rot),
                           EndEffectorConstraintSet(constraint_sk, frames, pos, rot, None,
                               joint_names=["LeftHand", "RightHand", "LeftFoot", "RightFoot", "Hips"])]
            first_heading = float(compute_heading_angle(pos, constraint_sk)[0])
            counts[0] += k  # Conditioning frames are EXTRA, not deducted from requested motion.
        if request.get("constraints"):
            from .authoring import compile_constraints
            from kimodo.constraints import load_constraints_lst
            records = request["constraints"]
            if source is None and request.get("align_first", True):
                from .motion_math import C
                initial = next((r for r in records if r.get("enabled", True) and
                    r["start_frame"] == int(request["sequence_origin"]) and r["type"] in {"fullbody", "root2d"}), None)
                if initial:
                    p = initial["payload"]
                    offset = C.T @ np.asarray(initial.get("translation", [0,0,0]), dtype=float)
                    if initial["type"] == "root2d":
                        origin[[0,2]] = np.asarray(p["smooth_root_2d"][0]) + offset[[0,2]]
                        if "global_root_heading" in p:
                            first_heading = float(np.arctan2(p["global_root_heading"][0][1], p["global_root_heading"][0][0]))
                    else:
                        origin = np.asarray(p["root_positions"][0], dtype=np.float32) + offset
                        origin[1] = 0.
                        local = self.tensor(np.repeat(p["local_rot_mats"][:1], 2, axis=0), train)
                        if train.nbjoints == 30 and local.shape[1] == 77:
                            local = train.from_SOMASkeleton77(local)
                        root = self.tensor(np.repeat(p["root_positions"][:1], 2, axis=0), train)
                        d = self.complete(local, root, train, fps)
                        first_heading = float(compute_heading_angle(d["posed_joints"], train)[0])
            expected = self.metadata(out_skeleton)
            for record in records:
                meta = record.get("skeleton")
                if meta and (meta.get("names") != expected["names"] or meta.get("parents") != expected["parents"]):
                    raise ValueError("Authored constraint rig differs from the selected model skeleton")
            native = compile_constraints(records, float(request["scene_fps"]), fps,
                int(request["sequence_origin"]), sum(counts)-k, k, origin)
            constraints.extend(load_constraints_lst(native, self.build(train.nbjoints)))
        status("generating", "Generating motion; see worker.log for denoising output")
        post = bool(request.get("postprocess", False)) and "g1" not in out_skeleton.name.lower()
        with self.torch.inference_mode():
            output = model([p.text for p in prompts], counts, num_denoising_steps=steps,
                           num_samples=1, multi_prompt=True, constraint_lst=constraints,
                           num_transition_frames=transition, first_heading_angle=first_heading,
                           cfg_weight=[float(request.get("text_guidance", 2.)),
                                       float(request.get("constraint_guidance", 2.))],
                           post_processing=post, root_margin=float(request.get("root_margin", .04)), return_numpy=True)
        generated = Clip(_single(output["local_rot_mats"], 4),
                         _single(output["root_positions"], 2), fps).validate()
        if len(generated.root) != sum(counts):
            raise RuntimeError(f"Kimodo returned {len(generated.root)} frames, expected {sum(counts)}. "
                               "Upstream timing API changed; refusing silent timeline drift")
        generated.root += origin
        output_fps = float(request.get("output_fps", fps))
        if source is None:
            total = frame_prompts(prompts, output_fps)[-1].end
            result = resample(generated, output_fps, count=total)
        else:
            # The visible prefix is sampled directly from the ORIGINAL, not round-tripped
            # through model FPS. At matching source/output FPS it remains byte-equivalent.
            prefix = resample(source, output_fps)
            suffix = Clip(generated.local[k:], generated.root[k:], fps)
            total = frame_prompts(prompts, output_fps)[-1].end
            suffix = resample(suffix, output_fps, count=total)
            blend = int(request.get("blend_frames", 0))
            if blend:
                if not 2 <= blend <= k:
                    raise ValueError("Blend frames must be 0 or between 2 and context_frames")
                # Align real timestamps of the overlapping context, ending one output sample
                # before the new segment. This keeps source and generated clocks independent.
                context_count = min(len(prefix.root), max(2, round_frame(k * output_fps/fps)))
                sample_times = k - (context_count - np.arange(context_count))*fps/output_fps
                from .motion_math import matrix_to_quat, quat_to_matrix, slerp
                t = np.clip(sample_times, 0., k-1)
                lo = np.floor(t).astype(int); hi = np.minimum(lo+1, k-1); a = t-lo
                q = matrix_to_quat(generated.local[:k])
                context_local = quat_to_matrix(slerp(q[lo], q[hi], a[:, None, None]))
                context_root = (1-a[:, None])*generated.root[lo] + a[:, None]*generated.root[hi]
                g = Clip(np.concatenate([context_local, suffix.local]),
                         np.concatenate([context_root, suffix.root]), output_fps)
                out_blend = min(context_count, max(2, round_frame(blend*output_fps/fps)))
                result = join_context(prefix, g, context_count, out_blend)
            else:
                result = Clip(np.concatenate([prefix.local, suffix.local]),
                              np.concatenate([prefix.root, suffix.root]), output_fps).validate()
        return result, out_skeleton, {"model_fps": fps, "context_frames": k,
                                      "added_frames": frame_prompts(prompts, output_fps)[-1].end,
                                      "prefix_preserved": not bool(request.get("blend_frames", 0))}


def run_job(job_dir):
    folder = Path(job_dir).resolve()
    request = load_json(folder / "request.json")
    if request.get("schema_version") != 1:
        raise ValueError("Unsupported worker request schema")
    operation = request.get("operation")
    if operation not in {"check", "import", "generate", "continue", "export", "prepare"} | STORAGE_OPERATIONS:
        raise ValueError("Unknown worker operation")
    def status(state, message):
        atomic_json(folder / "status.json", {"state": state, "message": message})
    configure_environment(request)
    if operation in STORAGE_OPERATIONS:
        if operation in DOWNLOAD_OPERATIONS:
            download_assets(request, status)
        report = inspect_storage(request)
        atomic_json(folder / "model_report.json", report)
        message = ("Required local files are available (not an inference test)" if report["ready"]
                   else "Local file check finished; some model/dependency files are missing. See model_report.json")
        result = {"state": "done", "message": message, "model_report": report}
        atomic_json(folder / "status.json", result)
        return result
    if "storage" in request and operation in {"generate", "continue", "prepare"}:
        report = inspect_storage(request)
        if not report["ready"]:
            missing = [row["component"] for row in report["components"] if not row["ready"]]
            atomic_json(folder / "model_report.json", report)
            raise ValueError("Missing/incomplete local files: " + "; ".join(missing) +
                             ". Use Check local files and Download / resume, or fix the manual path")
    status("initializing", "Checking Kimodo runtime")
    adapter = KimodoAdapter()
    if operation == "check":
        import kimodo
        info = {"state": "done", "message": "Kimodo import succeeded (model weights not tested)",
                "kimodo_file": str(kimodo.__file__), "torch": adapter.torch.__version__,
                "cuda_available": adapter.torch.cuda.is_available()}
        atomic_json(folder / "status.json", info)
        return info
    count = int(request.get("num_samples", 1)) if operation in {"generate", "continue"} else 1
    if not 1 <= count <= 8:
        raise ValueError("Choose 1–8 samples per job")
    samples = []
    for i in range(count):
        sample_request = dict(request)
        if int(request.get("seed", -1)) >= 0:
            sample_request["seed"] = (int(request["seed"]) + i) % (2**31)
        def sample_status(state, message):
            status(state, f"Sample {i+1}/{count}: {message}")
        if operation in {"import", "export"}:
            clip, skeleton = adapter.source(request["source"], request.get("source_fps"))
            clip = resample(clip, float(request.get("output_fps", clip.fps)))
            info = {}
        elif operation == "prepare":
            model = load_selected_model(request)
            skeleton = adapter.build(77 if model.skeleton.nbjoints == 30 else model.skeleton.nbjoints)
            neutral_frames = max(30, int(round(float(request.get("output_fps", model.fps)))))
            root = np.zeros((neutral_frames,3), dtype=np.float32)
            # Neutral joints have pelvis height; a rest rig should stand on the ground.
            root[:,1] = float(_numpy(skeleton.neutral_joints)[skeleton.root_idx,1])
            clip = Clip(np.tile(np.eye(3),(neutral_frames,skeleton.nbjoints,1,1)), root,
                        float(request.get("output_fps", model.fps))).validate()
            info = {"model_fps": float(model.fps)}
        else:
            clip, skeleton, info = adapter.generate(sample_request, sample_status)
        sample_status("saving", "Rebuilding FK/contacts and writing motion files")
        target = folder if count == 1 else folder / f"sample_{i+1:02d}"
        target.mkdir(exist_ok=True)
        files = adapter.save(target, clip, skeleton, export_bvh=bool(request.get("export_bvh", True)))
        if operation in {"generate", "continue"}:
            atomic_json(target / "meta.json", {**request["timeline"], "seed": sample_request.get("seed"),
                "model": request.get("model"), "continuation": operation == "continue"})
        samples.append({"directory": "." if count==1 else target.name,
                        "seed": sample_request.get("seed", -1), "files": files,
                        "frames": len(clip.root), "fps": clip.fps, "skeleton": skeleton.name, **info})
    result = {"state": "done", "message": "Model rig prepared (weights reload on the next job)" if operation=="prepare" else "Motion ready",
              "samples": samples, **samples[0]}
    atomic_json(folder / "status.json", result)
    return result
