import copy
import json
import numpy as np
import pytest
from kimodo_motion_studio.authoring import (compile_constraints, validate_record, native_to_records,
    model_name, MODEL_PRESETS, resize_duration, normalize_project, load_authoring_json)
from kimodo_motion_studio.motion_math import forward_native, global_to_model_local


def point(kind='root2d',start=1,end=1):
    payload={'smooth_root_2d':[[2.,3.]]} if kind=='root2d' else {
        'local_rot_mats':np.tile(np.eye(3),(1,3,1,1)).tolist(),'root_positions':[[2.,1.,3.]]}
    return {'type':kind,'start_frame':start,'end_frame':end,'payload':payload}


@pytest.mark.parametrize('preset',MODEL_PRESETS)
def test_model_presets(preset):
    assert model_name(*preset).startswith('Kimodo-')


def test_invalid_model():
    with pytest.raises(ValueError): model_name('SEED','SMPLX','v1.1')


def test_keyframe_conversion_and_context_offset():
    native=compile_constraints([point(start=49,end=49)],24.,30.,1,180,12,world_shift=[1,0,1])
    assert native[0]['frame_indices']==[72]
    assert native[0]['smooth_root_2d']==[[1.,2.]]


def test_inclusive_interval_retime():
    native=compile_constraints([point('fullbody',1,25)],24.,30.,1,90)
    assert native[0]['frame_indices']==list(range(31))
    assert len(native[0]['local_rot_mats'])==31


@pytest.mark.parametrize('start,end',[(0,1),(1,100),(25,24)])
def test_out_of_range(start,end):
    with pytest.raises(ValueError): compile_constraints([point(start=start,end=end)],30.,30.,1,30)


def test_disabled_does_not_restrict_range():
    r=point(start=-4,end=-4);r['enabled']=False
    assert compile_constraints([r],30,30,1,60)==[]


def test_same_track_overlap_rejected():
    with pytest.raises(ValueError,match='Overlapping'): compile_constraints([point(),point()],30,30,1,30)


def test_distinct_track_overlap_allowed():
    assert len(compile_constraints([point(),point('fullbody')],30,30,1,30))==2


def test_translation_blender_to_kimodo():
    r=point('fullbody');r['translation']=[1,2,3]
    d=compile_constraints([r],30,30,1,30)[0]
    assert d['root_positions']==[[3.,4.,1.]]


def test_sparse_import_splits_without_filling_gap():
    data=[{'type':'root2d','frame_indices':[0,2,3], 'smooth_root_2d':[[0,0],[2,0],[3,0]]}]
    records=native_to_records(data,30,30,100)
    assert [(r['start_frame'],r['end_frame']) for r in records]==[(100,100),(102,103)]
    native=compile_constraints(records,30,30,100,4)
    assert [d['frame_indices'] for d in native]==[[0],[2,3]]


@pytest.mark.parametrize('bad',[[-1],[True],[0,0],[2,1],[]])
def test_native_bad_indices(bad):
    with pytest.raises(ValueError): native_to_records([{'type':'root2d','frame_indices':bad,'smooth_root_2d':[[0,0]]}],30,30,1)


def test_import_single_fullbody_pose():
    r=point('fullbody');r=compile_constraints([r],30,30,1,30)
    records=native_to_records(r,30,24,1)
    assert records[0]['start_frame']==1


def test_invalid_rotation():
    r=point('fullbody');r['payload']['local_rot_mats'][0][0][0][0]=2.
    with pytest.raises(ValueError,match='scale/shear'): validate_record(r)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),'oops'])
def test_bad_root(bad):
    r=point();r['payload']['smooth_root_2d'][0][0]=bad
    with pytest.raises(ValueError):validate_record(r)


def test_heading_wrap_interpolation():
    r=point(end=3)
    angles=np.deg2rad([179,-179]);r['payload']['smooth_root_2d']=[[0,0],[0,0]]
    r['payload']['global_root_heading']=np.stack([np.cos(angles),np.sin(angles)],axis=-1).tolist()
    result=compile_constraints([r],30,30,1,90)[0]
    assert result['global_root_heading'][1][0]<-.999


@pytest.mark.parametrize('a,b,fps,expected',[(1,181,30,6.),(20,19,30,1/30),(0,24,24,1.)])
def test_prompt_edge_timing(a,b,fps,expected):
    assert resize_duration(a,b,fps)==pytest.approx(expected)


def project():
    return {'kind':'kimodo-studio-project','schema_version':2,'fps':30,
        'prompts':[{'text':'walk','duration':6,'enabled':False},{'text':'wave','duration':2}],
        'constraints':[point()], 'settings':{'seed':42,'python_path':'untrusted','allow_downloads':True}}


def test_project_roundtrip_mutes_and_no_backend_injection():
    p=normalize_project(project()); assert p['prompts'][0]['enabled'] is False
    assert p['settings']=={'seed':42}
    assert normalize_project(json.loads(json.dumps(p)))==p


def test_project_copies_not_mutates():
    data=project(); saved=copy.deepcopy(data); normalize_project(data);assert data==saved


@pytest.mark.parametrize('field,value',[('fps',0),('fps',float('nan')),('prompts',[]),('constraints','invalid')])
def test_bad_project(field,value):
    p=project();p[field]=value
    with pytest.raises(ValueError):normalize_project(p)


def test_nonfinite_file(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{"x":NaN}')
    with pytest.raises(ValueError): load_authoring_json(p)


def test_numpy_fk_matches_local_inverse():
    from kimodo_motion_studio.motion_math import quat_to_matrix
    angle=.6;q=np.array([np.cos(angle/2),0,0,np.sin(angle/2)])
    R=quat_to_matrix(q)
    meta={'parents':[-1,0,1],'rest_joints':[[0,0,0],[0,1,0],[0,2,0]],
          'rest_global_rot_mats':np.tile(np.eye(3),(3,1,1)).tolist()}
    local=np.tile(R,(2,3,1,1));root=np.array([[2,3,4],[3,4,5]])
    data=forward_native(local,root,meta)
    np.testing.assert_allclose(data['posed_joints'][:,0],root)
    recovered=global_to_model_local(data['global_rot_mats'],meta['parents'],meta['rest_global_rot_mats'])
    np.testing.assert_allclose(recovered,local,atol=1e-6)
