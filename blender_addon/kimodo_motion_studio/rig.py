# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender-only rig transfer. Does not import torch, scipy, or Kimodo."""
import json
import math
import numpy as np
import bpy
from mathutils import Matrix, Vector
from .motion_math import (C, Clip, load_portable, save_portable,
                          global_to_model_local, validate_sampled_offsets)


def scene_fps(scene):
    return scene.render.fps / scene.render.fps_base


def order(parents):
    result, remaining = [], set(range(len(parents)))
    while remaining:
        ready = sorted(i for i in remaining if parents[i] == -1 or parents[i] in result)
        if not ready:
            raise ValueError("Invalid/cyclic skeleton hierarchy")
        result.extend(ready); remaining.difference_update(ready)
    return result


def action_curves(action):
    """Blender 4.2 legacy and 4.4+/5.x slotted Actions."""
    if hasattr(action, "layers"):
        found = []
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, "channelbags"):
                    for bag in strip.channelbags:
                        found.extend(bag.fcurves)
        if found:
            return found
    return list(action.fcurves) if hasattr(action, "fcurves") else []


def bake_motion(path, scene, start_frame, name="Kimodo Motion"):
    """Generator: execute only from the main thread, advancing a few frames per timer.

    Always creates a new armature and Action. Closing before completion rolls it back.
    """
    clip, meta, arrays = load_portable(path)
    if "posed_joints" not in arrays or "global_rot_mats" not in arrays:
        raise ValueError("Interchange file lacks forward-kinematic pose channels")
    names, parents = meta["names"], meta["parents"]
    positions, rotations = arrays["posed_joints"], arrays["global_rot_mats"]
    if positions.shape != (len(clip.root), len(names), 3) or rotations.shape != clip.local.shape:
        raise ValueError("Interchange pose channels have inconsistent dimensions")
    if any(len(n) > 63 for n in names):
        raise ValueError("A skeleton bone name exceeds Blender's 63-character limit")
    if bpy.context.mode != "OBJECT":
        raise ValueError("Switch to Object Mode before importing motion")
    rest = np.asarray(meta["rest_joints"]) @ C.T
    rest_rot = C @ np.asarray(meta["rest_global_rot_mats"]) @ C.T
    arm = bpy.data.armatures.new(name + " Rig")
    obj = bpy.data.objects.new(name, arm)
    scene.collection.objects.link(obj)
    action = None
    succeeded = False
    saved_active = bpy.context.view_layer.objects.active
    saved_selected = list(bpy.context.selected_objects)
    saved_frame, saved_subframe = scene.frame_current, scene.frame_subframe
    try:
        for selected in saved_selected:
            selected.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        for i in order(parents):
            bone = arm.edit_bones.new(names[i])
            bone.head = Vector(rest[i])
            children = [j for j, p in enumerate(parents) if p == i and np.linalg.norm(rest[j]-rest[i]) > 0.001]
            direction = rest[children[0]] - rest[i] if children else np.array([0., 0., 0.055])
            bone.tail = Vector(rest[i] + direction)
            if parents[i] != -1:
                bone.parent = arm.edit_bones[names[parents[i]]]
            bone.use_connect = False
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.show_in_front = True
        arm.display_type = "STICK"
        obj["kimodo_skeleton_json"] = json.dumps(meta)
        obj["kimodo_motion_path"] = str(path)
        corrections = []
        for i, name_i in enumerate(names):
            b = arm.bones[name_i]
            correction = rest_rot[i].T @ np.asarray(b.matrix_local.to_3x3())
            corrections.append(correction)
            b["kimodo_basis_correction"] = correction.flatten().tolist()
            obj.pose.bones[name_i].rotation_mode = "QUATERNION"
        obj.animation_data_create()
        action = bpy.data.actions.new(name)
        obj.animation_data.action = action
        action.use_fake_user = True
        # keyframe_insert creates a compatible action slot when needed in modern Blender.
        sequence = order(parents)
        previous_quats = {}
        for f in range(len(clip.root)):
            targets = []
            for j in range(len(names)):
                r = C @ rotations[f, j] @ C.T @ corrections[j]
                m = Matrix(r.tolist()).to_4x4()
                m.translation = Vector(C @ positions[f, j])
                targets.append(m)
            frame = float(start_frame) + f * scene_fps(scene) / clip.fps
            for j in sequence:
                pb = obj.pose.bones[names[j]]
                rest_matrix = pb.bone.matrix_local
                parent = parents[j]
                if parent == -1:
                    basis = rest_matrix.inverted() @ targets[j]
                else:
                    parent_rest = arm.bones[names[parent]].matrix_local
                    basis = (parent_rest.inverted() @ rest_matrix).inverted() @ targets[parent].inverted() @ targets[j]
                loc, quat, scale = basis.decompose()
                if j in previous_quats and previous_quats[j].dot(quat) < 0:
                    quat.negate()
                previous_quats[j] = quat.copy()
                pb.location, pb.rotation_quaternion, pb.scale = loc, quat, scale
                pb.keyframe_insert("location", frame=frame, group=names[j])
                pb.keyframe_insert("rotation_quaternion", frame=frame, group=names[j])
                pb.keyframe_insert("scale", frame=frame, group=names[j])
            yield f + 1, len(clip.root), obj
        for curve in action_curves(action):
            for key in curve.keyframe_points:
                key.interpolation = "LINEAR"
        obj["kimodo_frame_start"] = float(start_frame)
        last_frame = float(start_frame) + (len(clip.root)-1) * scene_fps(scene)/clip.fps
        obj["kimodo_frame_end"] = last_frame
        scene.frame_end = max(scene.frame_end, math.ceil(last_frame))
        scene.frame_set(math.floor(start_frame), subframe=float(start_frame) % 1)
        succeeded = True
    finally:
        if not succeeded:
            if obj.mode != "OBJECT":
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.data.objects.remove(obj, do_unlink=True)
            if arm.users == 0:
                bpy.data.armatures.remove(arm)
            if action is not None and action.users <= 1:
                action.use_fake_user = False
                if action.users == 0:
                    bpy.data.actions.remove(action)
            for selected in saved_selected:
                if selected.name in bpy.data.objects:
                    selected.select_set(True)
            if saved_active is not None and saved_active.name in bpy.data.objects:
                bpy.context.view_layer.objects.active = saved_active
            scene.frame_set(saved_frame, subframe=saved_subframe)


def snapshot(obj, scene, start, end, path):
    """Sample an evaluated native rig, including its NLA/constraints and world transform.

    Retarget arbitrary rigs onto a Kimodo rig first. Metadata validation is deliberate;
    matching only bone counts cannot prove compatible bone ordering or rest orientation.
    """
    if obj is None or obj.type != "ARMATURE" or "kimodo_skeleton_json" not in obj:
        raise ValueError("Select a rig imported/generated by this add-on. Retarget other rigs onto it first")
    if end < start or end-start+1 < 2 or end-start+1 > 36000:
        raise ValueError("Choose a source range containing 2–36000 frames")
    meta = json.loads(obj["kimodo_skeleton_json"])
    names, parents = meta["names"], meta["parents"]
    if any(n not in obj.pose.bones for n in names):
        raise ValueError("Native rig bones have been renamed or removed")
    for j, n in enumerate(names):
        parent_name = obj.data.bones[n].parent.name if obj.data.bones[n].parent else None
        expected = names[parents[j]] if parents[j] != -1 else None
        if parent_name != expected:
            raise ValueError("Native rig hierarchy was edited; retarget to an unmodified Kimodo rig")
    correction = [np.asarray(obj.data.bones[n]["kimodo_basis_correction"]).reshape(3, 3) for n in names]
    rest_pos = np.asarray(meta["rest_joints"])
    rest_rot = np.asarray(meta["rest_global_rot_mats"])
    for j, n in enumerate(names):
        bone = obj.data.bones[n]
        expected_r = C @ rest_rot[j] @ C.T @ correction[j]
        if (not np.allclose(np.asarray(bone.head_local), C @ rest_pos[j], atol=0.001)
                or not np.allclose(np.asarray(bone.matrix_local.to_3x3()), expected_r, atol=0.001)):
            raise ValueError("Native rig rest pose was edited. Retarget to an unmodified Kimodo rig")
    rotations, roots = [], []
    saved_frame, saved_subframe = scene.frame_current, scene.frame_subframe
    try:
        for frame in range(start, end+1):
            scene.frame_set(frame)
            dep = bpy.context.evaluated_depsgraph_get()
            evaluated = obj.evaluated_get(dep)
            global_r, global_p = [], []
            for j, n in enumerate(names):
                pose = evaluated.matrix_world @ evaluated.pose.bones[n].matrix
                matrix = np.asarray(pose.to_3x3())
                if not np.allclose(matrix.T @ matrix, np.eye(3), atol=0.005) or np.linalg.det(matrix) < 0.99:
                    raise ValueError("Source rig has scale/shear. Apply or bake to a unit-scale native rig first")
                global_r.append(C.T @ matrix @ correction[j].T @ C)
                global_p.append(C.T @ np.array(pose.translation))
            validate_sampled_offsets(global_p, global_r, parents, rest_pos, rest_rot)
            local = global_to_model_local(global_r, parents, rest_rot)
            rotations.append(local)
            roots.append(global_p[meta["root_idx"]])
        clip = Clip(np.asarray(rotations), np.asarray(roots), scene_fps(scene)).validate()
        save_portable(path, clip, meta)
        return clip
    finally:
        scene.frame_set(saved_frame, subframe=saved_subframe)
