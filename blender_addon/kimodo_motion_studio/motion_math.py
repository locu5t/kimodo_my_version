# SPDX-License-Identifier: GPL-3.0-or-later
"""NumPy-only motion operations; quaternion ordering is always w,x,y,z."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from .timeline import round_frame

C = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])  # Y-up -> Z-up
MAX_NPZ_BYTES = 1024 * 1024 * 1024


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-10) or not np.all(np.isfinite(q)):
        raise ValueError("Invalid zero or non-finite quaternion")
    return q / n


def slerp(q0, q1, alpha):
    q0, q1 = normalize_quat(q0), normalize_quat(q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0., 1.)
    a = np.asarray(alpha, dtype=np.float64)
    while a.ndim < q0.ndim:
        a = a[..., None]
    theta = np.arccos(dot)
    denom = np.maximum(np.sin(theta), 1e-10)
    spherical = np.sin((1 - a) * theta) / denom * q0 + np.sin(a * theta) / denom * q1
    linear = (1 - a) * q0 + a * q1
    return normalize_quat(np.where(dot > 0.9995, linear, spherical))


def quat_to_matrix(q):
    w, x, y, z = np.moveaxis(normalize_quat(q), -1, 0)
    return np.stack([
        1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y),
    ], axis=-1).reshape(w.shape + (3, 3))


def matrix_to_quat(matrix):
    """Largest-diagonal branch stays stable for rotations close to 180 degrees."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Expected (...,3,3) rotations")
    flat = matrix.reshape(-1, 3, 3)
    result = np.empty((len(flat), 4))
    for i, m in enumerate(flat):
        tr = np.trace(m)
        if tr > 0:
            s = math.sqrt(max(1. + tr, 0.)) * 2
            result[i] = [s/4, (m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s]
        else:
            k = int(np.argmax(np.diag(m)))
            j, l = (k+1) % 3, (k+2) % 3
            s = math.sqrt(max(1 + m[k,k] - m[j,j] - m[l,l], 0.)) * 2
            if s < 1e-10:
                raise ValueError("Invalid rotation matrix")
            q = np.zeros(4)
            q[0] = (m[l,j] - m[j,l])/s
            q[k+1] = s/4
            q[j+1] = (m[j,k] + m[k,j])/s
            q[l+1] = (m[l,k] + m[k,l])/s
            result[i] = q
    return normalize_quat(result.reshape(matrix.shape[:-2] + (4,)))


@dataclass
class Clip:
    local: np.ndarray
    root: np.ndarray
    fps: float

    def validate(self):
        self.local = np.asarray(self.local, dtype=np.float32)
        self.root = np.asarray(self.root, dtype=np.float32)
        self.fps = float(self.fps)
        if self.local.ndim != 4 or self.local.shape[-2:] != (3, 3):
            raise ValueError("local_rot_mats must have shape (T,J,3,3)")
        if self.root.shape != (len(self.local), 3) or len(self.local) < 2:
            raise ValueError("root_positions must have shape (T,3), with at least two frames")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("FPS must be finite and positive")
        if not np.isfinite(self.local).all() or not np.isfinite(self.root).all():
            raise ValueError("Motion contains non-finite values")
        eye = np.eye(3)
        if not np.allclose(np.swapaxes(self.local, -1, -2) @ self.local, eye, atol=0.015):
            raise ValueError("Joint matrices contain scale/shear rather than rotations")
        if np.any(np.linalg.det(self.local) < 0.98):
            raise ValueError("Joint matrices must be proper rotations")
        return self

    def copy(self):
        return Clip(self.local.copy(), self.root.copy(), self.fps)


def resample(clip, fps, count=None):
    clip.validate()
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Target FPS must be positive and finite")
    n = round_frame(len(clip.root) * fps / clip.fps) if count is None else count
    if n < 2:
        raise ValueError("Resampling would produce fewer than two frames")
    if abs(clip.fps - fps) < 1e-9 and n == len(clip.root):
        return clip.copy()
    # Sample actual times. Do not stretch the entire clip with linspace endpoints.
    t = np.minimum(np.arange(n) * clip.fps / fps, len(clip.root) - 1)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(clip.root) - 1)
    a = t - lo
    root = (1-a[:, None])*clip.root[lo] + a[:, None]*clip.root[hi]
    q = matrix_to_quat(clip.local)
    return Clip(quat_to_matrix(slerp(q[lo], q[hi], a[:, None, None])), root, fps).validate()


def join_context(prefix, generated, context_frames, blend_frames=0):
    """Generated[0:K] occupies the SAME time as prefix[-K:]. Drop it once.

    blend=0 preserves every prefix sample. blend>0 changes only the final B
    source samples in this NEW clip; callers retain their original Action.
    Output duration is P + G - K samples, not P + G.
    """
    prefix.validate(); generated.validate()
    if prefix.local.shape[1:] != generated.local.shape[1:] or abs(prefix.fps-generated.fps) > 1e-8:
        raise ValueError("Join requires matching skeleton shapes and FPS")
    k, b = int(context_frames), int(blend_frames)
    if not 1 <= k <= min(len(prefix.root), len(generated.root)-1):
        raise ValueError("Invalid continuation context length")
    if not 0 <= b <= k or b == 1:
        raise ValueError("Blend must be 0 or between 2 and context_frames")
    result = prefix.copy()
    if b:
        t = np.linspace(0., 1., b)
        a = t*t*(3.-2.*t)
        result.root[-b:] = (1-a[:, None])*prefix.root[-b:] + a[:, None]*generated.root[k-b:k]
        result.local[-b:] = quat_to_matrix(slerp(
            matrix_to_quat(prefix.local[-b:]), matrix_to_quat(generated.local[k-b:k]), a[:, None, None]))
    return Clip(np.concatenate([result.local, generated.local[k:]]),
                np.concatenate([result.root, generated.root[k:]]), prefix.fps).validate()


def safe_npz(path):
    """Reject pickle-backed arrays and oversized ZIPs before using upstream import."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > MAX_NPZ_BYTES:
            raise ValueError("NPZ uncompressed size exceeds 1 GiB")
    with np.load(path, allow_pickle=False) as z:
        # Eager loading is intentional: validates every member, including metadata.
        return {k: np.array(z[k]) for k in z.files}


def load_portable(path):
    d = safe_npz(path)
    if int(d.get("kimodo_blender_schema", 0)) != 1:
        raise ValueError("Not a Kimodo Blender interchange file")
    clip = Clip(d["local_rot_mats"], d["root_positions"], float(d["fps"])).validate()
    meta = json.loads(str(d["skeleton_json"].item()))
    j = clip.local.shape[1]
    if len(meta["names"]) != j or len(meta["parents"]) != j or len(set(meta["names"])) != j:
        raise ValueError("Invalid skeleton metadata")
    parents = meta["parents"]
    if sum(p == -1 for p in parents) != 1 or parents.index(-1) != meta["root_idx"]:
        raise ValueError("Skeleton must contain one identified root")
    for i, p in enumerate(parents):
        if not isinstance(p, int) or not -1 <= p < j or p == i:
            raise ValueError("Invalid skeleton parent")
        seen = {i}
        while p != -1:
            if p in seen:
                raise ValueError("Cyclic skeleton hierarchy")
            seen.add(p); p = parents[p]
    return clip, meta, d


def save_portable(path, clip, skeleton_meta, derived=None):
    clip.validate()
    d = dict(derived or {})
    d.update(local_rot_mats=clip.local, root_positions=clip.root, fps=np.float64(clip.fps),
             kimodo_blender_schema=np.int32(1), skeleton_json=np.array(json.dumps(skeleton_meta)))
    np.savez_compressed(path, **d)


def global_to_model_local(global_rotations, parents, rest_global_rotations):
    """Invert FK, including skeletons with baked local rest rotations (e.g. G1)."""
    g = np.asarray(global_rotations)
    rest = np.asarray(rest_global_rotations)
    n = len(parents)
    if g.shape[-3:] != (n, 3, 3) or rest.shape != (n, 3, 3):
        raise ValueError("Rotation channels do not match skeleton hierarchy")
    out = np.empty_like(g)
    for j, p in enumerate(parents):
        effective = g[..., j, :, :] if p == -1 else np.swapaxes(g[..., p, :, :], -1, -2) @ g[..., j, :, :]
        rest_local = rest[j] if p == -1 else rest[p].T @ rest[j]
        out[..., j, :, :] = rest_local.T @ effective
    return out


def validate_sampled_offsets(positions, global_rotations, parents, rest_positions, rest_rotations, tolerance=0.002):
    """Reject bone translation/stretch that fixed-bone-length Kimodo cannot encode."""
    pos, rot = np.asarray(positions), np.asarray(global_rotations)
    rest_pos, rest_rot = np.asarray(rest_positions), np.asarray(rest_rotations)
    for j, p in enumerate(parents):
        if p == -1:
            continue
        neutral_offset = rest_rot[p].T @ (rest_pos[j] - rest_pos[p])
        expected = pos[..., p, :] + (rot[..., p, :, :] @ neutral_offset[..., None])[..., 0]
        if not np.allclose(expected, pos[..., j, :], atol=tolerance, rtol=0):
            raise ValueError("Source has translated/stretched bones or edited proportions. Retarget onto an unmodified Kimodo rig")


def forward_native(local, root, meta):
    """NumPy FK from transfer metadata, including baked rest orientations."""
    local=np.asarray(local); root=np.asarray(root)
    rest=np.asarray(meta['rest_joints']); rest_r=np.asarray(meta['rest_global_rot_mats'])
    parents=meta['parents']; count=len(parents)
    rotations=np.empty_like(local); positions=np.empty((len(local),count,3))
    pending=set(range(count)); done=set()
    while pending:
        ready=[j for j in pending if parents[j]==-1 or parents[j] in done]
        if not ready:
            raise ValueError('Cyclic skeleton')
        for j in ready:
            p=parents[j]
            if p==-1:
                rotations[:,j]=rest_r[j] @ local[:,j]
                positions[:,j]=root
            else:
                rotations[:,j]=rotations[:,p] @ (rest_r[p].T @ rest_r[j]) @ local[:,j]
                offset=rest_r[p].T @ (rest[j]-rest[p])
                positions[:,j]=positions[:,p]+(rotations[:,p] @ offset[:,None])[...,0]
        pending.difference_update(ready); done.update(ready)
    return {'posed_joints':positions,'global_rot_mats':rotations}
