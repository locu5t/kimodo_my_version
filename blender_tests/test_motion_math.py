import numpy as np
import pytest
from kimodo_motion_studio.motion_math import (Clip, slerp, matrix_to_quat, quat_to_matrix,
    resample, join_context, save_portable, load_portable, safe_npz, C)


def clip(n=30, fps=30., joints=3):
    return Clip(np.tile(np.eye(3),(n,joints,1,1)), np.stack([np.arange(n)/fps,np.ones(n),np.zeros(n)],axis=-1), fps).validate()

@pytest.mark.parametrize("q", [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0.5,0.5,0.5,0.5]])
def test_identical_and_antipodal_slerp(q):
    q = np.asarray(q,dtype=float)
    for sign in (1,-1):
        for a in (0,0.1,0.5,0.9,1):
            assert np.allclose(slerp(q,sign*q,a),q)


def test_random_rotation_roundtrip():
    rng = np.random.default_rng(4)
    q = rng.normal(size=(6,3,4))
    r = quat_to_matrix(q)
    assert np.allclose(quat_to_matrix(matrix_to_quat(r)),r,atol=1e-7)
    assert np.allclose(np.linalg.det(r),1)


def test_half_turn_interpolation():
    q = slerp([1,0,0,0],[0,0,1,0],0.5)
    assert np.allclose(q, [2**-0.5,0,2**-0.5,0])

@pytest.mark.parametrize("target", [24,30,60,29.97])
def test_resample_stationary_rotations_stay_valid(target):
    result = resample(clip(120,60),target)
    assert np.isfinite(result.local).all()
    assert np.allclose(result.local,np.eye(3))
    assert len(result.root) == round(2*target)
    result.validate()


def test_resample_uses_real_timestamps():
    result = resample(clip(60,30),60)
    assert np.isclose(result.root[1,0],1/60)
    assert np.isclose(result.root[60,0],1)


def test_preserve_prefix_exactly():
    p, g = clip(180),clip(192)
    p.root[:,0] += 100
    g.root[:,0] += 105.6
    out = join_context(p,g,12,0)
    assert len(out.root) == 360  # 180 old + 180 new; 12 context frames not counted twice.
    np.testing.assert_array_equal(out.root[:180],p.root)
    np.testing.assert_array_equal(out.local[:180],p.local)
    np.testing.assert_array_equal(out.root[180:],g.root[12:])


def test_blend_modifies_only_chosen_tail_and_uses_slerp():
    p,g=clip(30),clip(22)
    g.root[:,0] += 8
    g.local[:] = quat_to_matrix([0,0,1,0])
    out=join_context(p,g,12,6)
    np.testing.assert_array_equal(out.root[:24],p.root[:24])
    np.testing.assert_array_equal(out.local[:24],p.local[:24])
    assert np.allclose(out.local[29],g.local[11])
    assert np.allclose(out.root[29],g.root[11])
    out.validate()

@pytest.mark.parametrize("context,blend", [(0,0),(40,0),(12,1),(12,13),(12,-1)])
def test_bad_join(context,blend):
    with pytest.raises(ValueError):
        join_context(clip(30),clip(30),context,blend)


def test_axis_rotation_is_proper_and_roundtrips():
    assert np.isclose(np.linalg.det(C),1.)
    assert np.allclose(C @ [0,1,0],[0,0,1])
    assert np.allclose(C.T @ C,np.eye(3))


def test_portable(tmp_path):
    file = tmp_path/"test.npz"
    meta={"names":["root","a","b"],"parents":[-1,0,1],"root_idx":0}
    p=clip()
    save_portable(file,p,meta)
    other,loaded,_=load_portable(file)
    np.testing.assert_array_equal(other.root,p.root)
    assert loaded==meta
    meta["parents"]=[-1,2,1]
    save_portable(file,p,meta)
    with pytest.raises(ValueError,match="Cyclic"):
        load_portable(file)


def test_object_npz_rejected(tmp_path):
    path=tmp_path/"unsafe.npz"
    np.savez(path,thing=np.array([{}],dtype=object))
    with pytest.raises(ValueError):
        safe_npz(path)


def test_nonrotation_rejected():
    p=clip()
    p.local[0,0,0,0]=2
    with pytest.raises(ValueError,match="scale/shear"):
        p.validate()


def test_bad_quaternion_rejected():
    with pytest.raises(ValueError):
        slerp([0,0,0,0],[1,0,0,0],0.5)


def test_baked_rest_global_to_model_local_roundtrip():
    from kimodo_motion_studio.motion_math import global_to_model_local, quat_to_matrix
    rng = np.random.default_rng(38)
    parents = [-1, 0, 1, 0]
    rest_local = quat_to_matrix(rng.normal(size=(4, 4)))
    model_local = quat_to_matrix(rng.normal(size=(7, 4, 4)))
    rest_global = np.empty_like(rest_local)
    global_r = np.empty_like(model_local)
    for j, p in enumerate(parents):
        rest_global[j] = rest_local[j] if p == -1 else rest_global[p] @ rest_local[j]
        effective = rest_local[j] @ model_local[:, j]
        global_r[:, j] = effective if p == -1 else global_r[:, p] @ effective
    np.testing.assert_allclose(global_to_model_local(global_r, parents, rest_global), model_local, atol=1e-6)


def test_baked_rest_identity_recovered_as_identity():
    from kimodo_motion_studio.motion_math import global_to_model_local, quat_to_matrix
    rest = quat_to_matrix(np.random.default_rng(16).normal(size=(3, 4)))
    result = global_to_model_local(rest, [-1, 0, 1], rest)
    np.testing.assert_allclose(result, np.tile(np.eye(3), (3, 1, 1)), atol=1e-6)


def test_fixed_bone_lengths_reject_translated_child():
    from kimodo_motion_studio.motion_math import validate_sampled_offsets
    parents = [-1, 0]
    rest = np.array([[0., 0., 0.], [0., 1., 0.]])
    rotations = np.tile(np.eye(3), (2, 1, 1))
    validate_sampled_offsets(rest, rotations, parents, rest, rotations)
    changed = rest.copy(); changed[1, 0] = 0.1
    with pytest.raises(ValueError, match="translated/stretched"):
        validate_sampled_offsets(changed, rotations, parents, rest, rotations)
