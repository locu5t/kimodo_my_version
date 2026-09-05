# SPDX-License-Identifier: GPL-3.0-or-later
"""UI-independent constraint/timeline contracts. Scene frames are inclusive.

Prompt spans stay half-open in timeline.py. Constraint intervals include BOTH
keyframes; this module is the sole scene-frame -> model-frame conversion point.
"""
import copy
import json
import math
from pathlib import Path
import numpy as np
from .timeline import frame_prompts, parse_prompts, round_frame
from .motion_math import C, Clip, matrix_to_quat, quat_to_matrix, slerp

TRACKS = (('fullbody', 'Full Body'), ('root2d', '2D Root'),
          ('left-hand', 'Left Hand'), ('right-hand', 'Right Hand'),
          ('left-foot', 'Left Foot'), ('right-foot', 'Right Foot'))
KINDS = {key for key, _ in TRACKS} | {'end-effector'}
MODEL_PRESETS = (
    ('RP', 'SOMA', 'v1'), ('RP', 'SOMA', 'v1.1'), ('RP', 'SMPLX', 'v1'),
    ('RP', 'G1', 'v1'), ('SEED', 'SOMA', 'v1'), ('SEED', 'SOMA', 'v1.1'), ('SEED', 'G1', 'v1'))
MAX_CONSTRAINTS = 512
MAX_CONSTRAINT_SAMPLES = 36000
MAX_PROJECT_BYTES = 32 * 1024 * 1024


def model_name(dataset, skeleton, version):
    if (dataset, skeleton, version) not in MODEL_PRESETS:
        raise ValueError('No released model for this dataset/skeleton/version; choose v1 or another dataset')
    return f'Kimodo-{skeleton}-{dataset}-{version}'


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{label} must be an integer')
    return value


def _array(value, shape_tail, label):
    try:
        a = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Invalid {label}') from exc
    if a.ndim != len(shape_tail)+1 or a.shape[1:] != tuple(shape_tail) or not 1 <= len(a) <= MAX_CONSTRAINT_SAMPLES:
        raise ValueError(f'{label} must be (T,{",".join(map(str, shape_tail))}), with 1–36000 samples')
    if not np.isfinite(a).all():
        raise ValueError(f'{label} contains NaN or infinity')
    return a


def validate_record(record):
    if not isinstance(record, dict) or record.get('type') not in KINDS:
        raise ValueError('Unknown constraint type')
    a = integer(record.get('start_frame'), 'Constraint start')
    b = integer(record.get('end_frame'), 'Constraint end')
    if b < a or b-a+1 > MAX_CONSTRAINT_SAMPLES:
        raise ValueError('Constraint interval must contain 1–36000 scene frames')
    if not isinstance(record.get('enabled', True), bool):
        raise ValueError('Constraint enabled must be boolean')
    p = record.get('payload')
    if not isinstance(p, dict):
        raise ValueError('Constraint needs a captured pose or waypoint payload')
    if record['type'] == 'root2d':
        root = _array(p.get('smooth_root_2d'), (2,), '2D root')
        if 'global_root_heading' in p:
            heading = _array(p['global_root_heading'], (2,), 'Heading')
            if len(heading) != len(root) or not np.allclose(np.linalg.norm(heading, axis=-1), 1., atol=.01):
                raise ValueError('Heading must contain matching unit (cos,sin) vectors')
    else:
        local = np.asarray(p.get('local_rot_mats'), dtype=np.float64)
        if local.ndim != 4 or local.shape[-2:] != (3,3) or not 1 <= local.shape[1] <= 256:
            raise ValueError('Captured local rotations must be (T,J,3,3)')
        roots = _array(p.get('root_positions'), (3,), 'Root positions')
        if len(local) != len(roots):
            raise ValueError('Captured rotations and roots have different sample counts')
        # Clip validates rigid matrices, including the one-sample keyframe case.
        Clip(np.repeat(local, 2, axis=0) if len(local)==1 else local,
             np.repeat(roots, 2, axis=0) if len(roots)==1 else roots, 30.).validate()
        if record['type'] == 'end-effector':
            joints = p.get('joint_names')
            if not isinstance(joints, list) or not joints or any(j not in {'LeftHand','RightHand','LeftFoot','RightFoot','Hips'} for j in joints):
                raise ValueError('Invalid end-effector joint_names')
    translation = np.asarray(record.get('translation', [0,0,0]), dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError('Constraint translation must be a finite Blender XYZ vector')
    return copy.deepcopy(record)


def _samples(a, count, rotation=False):
    a = np.asarray(a, dtype=float)
    t = np.linspace(0., len(a)-1, count)
    lo = np.floor(t).astype(int); hi = np.minimum(lo+1, len(a)-1)
    w = (t-lo).reshape((count,) + (1,)*(a.ndim-1))
    if rotation:
        q = matrix_to_quat(a)
        return quat_to_matrix(slerp(q[lo], q[hi], (t-lo)[:,None,None]))
    return (1-w)*a[lo] + w*a[hi]


def compile_constraints(records, scene_fps, model_fps, origin, total_frames,
                        context_frames=0, world_shift=(0.,0.,0.)):
    """Build native Kimodo dicts. Reject out-of-range constraints; never drop them.

    For continuation, constraints are placed AFTER the extra context and translated
    into the same local horizontal frame as that context. No duration is lost.
    """
    if not (math.isfinite(scene_fps) and math.isfinite(model_fps) and scene_fps > 0 and model_fps > 0):
        raise ValueError('Constraint FPS must be positive and finite')
    if len(records) > MAX_CONSTRAINTS:
        raise ValueError('Too many constraints')
    result = []
    occupied = set()
    for raw in records:
        r = validate_record(raw)
        if not r.get('enabled', True):
            continue
        first = round_frame((r['start_frame']-origin)*model_fps/scene_fps)
        last = round_frame((r['end_frame']-origin)*model_fps/scene_fps)
        if first < 0 or last >= total_frames or last < first:
            raise ValueError(f"{r.get('name',r['type'])}: constraint is outside the generated prompt range")
        indices = list(range(first+context_frames, last+context_frames+1))
        # Same-track overlaps are ambiguous and rejected, rather than last-write-wins.
        keys = {(r['type'], f) for f in indices}
        if keys & occupied:
            raise ValueError('Overlapping constraints on the same track; mute or move one')
        occupied |= keys
        p = r['payload']; n = len(indices)
        d = {'type': r['type'], 'frame_indices': indices}
        delta = C.T @ np.asarray(r.get('translation', [0,0,0]), dtype=float) - np.asarray(world_shift)
        if r['type'] == 'root2d':
            d['smooth_root_2d'] = (_samples(p['smooth_root_2d'], n) + delta[[0,2]]).tolist()
            if 'global_root_heading' in p:
                h = np.asarray(p['global_root_heading'])
                angles = np.unwrap(np.arctan2(h[:,1], h[:,0]))
                a = np.interp(np.linspace(0,len(h)-1,n), np.arange(len(h)), angles)
                d['global_root_heading'] = np.stack([np.cos(a),np.sin(a)],axis=-1).tolist()
        else:
            d['local_rot_mats'] = _samples(p['local_rot_mats'], n, rotation=True).tolist()
            d['root_positions'] = (_samples(p['root_positions'], n) + delta).tolist()
            if r['type'] == 'end-effector':
                d['joint_names'] = list(p['joint_names'])
        result.append(d)
    return result


def native_to_records(data, source_fps, scene_fps, origin):
    """Import native constraints.json. Split sparse keyframes into contiguous runs."""
    if not isinstance(data, list) or len(data) > MAX_CONSTRAINTS:
        raise ValueError('Native constraints.json must be an array')
    if min(source_fps, scene_fps) <= 0 or not all(map(math.isfinite,(source_fps,scene_fps))):
        raise ValueError('Invalid constraint import FPS')
    out=[]
    for entry in data:
        if not isinstance(entry, dict) or entry.get('type') not in KINDS:
            raise ValueError('Unknown native constraint type')
        frames=entry.get('frame_indices')
        if not isinstance(frames,list) or not frames:
            raise ValueError('Constraint has no frame_indices')
        if any(isinstance(f,bool) or not isinstance(f,int) or f<0 for f in frames) or frames != sorted(set(frames)):
            raise ValueError('Constraint frame indices must be sorted, unique non-negative integers')
        channels=('smooth_root_2d','global_root_heading') if entry['type']=='root2d' else ('local_rot_mats','root_positions')
        for key in channels:
            if key in entry and len(entry[key])!=len(frames):
                raise ValueError(f'{key} length does not match frame_indices')
        starts=[0]+[i for i in range(1,len(frames)) if frames[i]!=frames[i-1]+1]+[len(frames)]
        for a,b in zip(starts,starts[1:]):
            p={key:copy.deepcopy(entry[key][a:b]) for key in channels if key in entry}
            if 'joint_names' in entry:
                p['joint_names']=entry['joint_names']
            rec={'type':entry['type'],'name':entry['type'], 'enabled':True,
                 'start_frame':origin+round_frame(frames[a]*scene_fps/source_fps),
                 'end_frame':origin+round_frame(frames[b-1]*scene_fps/source_fps), 'payload':p}
            out.append(validate_record(rec))
    if len(out)>MAX_CONSTRAINTS:
        raise ValueError('Imported constraints exceed the 512-track-item limit')
    return out


def load_authoring_json(path):
    p=Path(path)
    if p.stat().st_size>MAX_PROJECT_BYTES:
        raise ValueError('Project/constraint JSON exceeds 32 MiB')
    def reject(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    return json.loads(p.read_text(encoding='utf-8-sig'),parse_constant=reject)


def resize_duration(start, new_end, fps):
    """Right-edge ripple trim: one-frame minimum, no gap introduced."""
    if not math.isfinite(fps) or fps<=0:
        raise ValueError('FPS must be finite and positive')
    return max(1,round_frame(new_end)-start)/fps


def normalize_project(data):
    if not isinstance(data,dict) or data.get('kind')!='kimodo-studio-project' or data.get('schema_version')!=2:
        raise ValueError('Not a Kimodo Studio v2 project JSON')
    fps=float(data.get('fps',0))
    if not math.isfinite(fps) or fps<=0:
        raise ValueError('Project FPS must be positive and finite')
    rows=data.get('prompts')
    parsed=parse_prompts(rows)
    frame_prompts(parsed,fps)
    if any(not isinstance(p.get('enabled',True),bool) for p in rows):
        raise ValueError('Prompt enabled flags must be boolean')
    constraints=data.get('constraints',[])
    if not isinstance(constraints,list) or len(constraints)>MAX_CONSTRAINTS:
        raise ValueError('Invalid project constraints')
    result=copy.deepcopy(data)
    result['constraints']=[validate_record(r) for r in constraints]
    if not isinstance(result.get('settings',{}),dict):
        raise ValueError('Project settings must be an object')
    # Never accept executable paths, downloads, or backend configuration from a project.
    allowed={'start_frame':int,'source_start':int,'source_end':int,'preview_append':bool,
             'steps':int,'seed':int,'transition':int,'context':int,'blend':int,
             'text_guidance':(int,float),'constraint_guidance':(int,float),
             'postprocess':bool,'num_samples':int,'model':str}
    result['settings']={k:v for k,v in result.get('settings',{}).items() if k in allowed}
    for k,v in result['settings'].items():
        expected=allowed[k]
        if not isinstance(v,expected) or (expected!=bool and isinstance(v,bool)):
            raise ValueError(f'Invalid project setting {k}')
        if isinstance(v,(int,float)) and not math.isfinite(v):
            raise ValueError(f'Non-finite setting {k}')
    return result
