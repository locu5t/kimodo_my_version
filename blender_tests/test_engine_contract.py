"""Fake-model contract tests, NOT GPU inference/quality tests."""
import sys
import types
import numpy as np
import pytest
from kimodo_motion_studio.motion_math import Clip
from kimodo_motion_studio.engine import KimodoAdapter

@pytest.fixture
def adapter(monkeypatch):
    torch=pytest.importorskip("torch")
    calls={}
    class Skeleton:
        nbjoints=3
        name="test_skeleton"
        neutral_joints=torch.zeros((3,3))
        root_idx=0
    sk=Skeleton()
    class Constraint:
        def __init__(self,skeleton,frames,pos,rot,*args,**kwargs):
            self.frames,self.pos,self.rot=frames,pos,rot
    class EE(Constraint):
        def __init__(self,skeleton,frames,pos,rot,smooth_root_2d,*,joint_names):
            assert smooth_root_2d is None
            assert "Hips" in joint_names
            super().__init__(skeleton,frames,pos,rot)
    def complete(local,root,skeleton,fps):
        pos=root[:,None,:].repeat(1,3,1)
        pos[:,1,0]+=1.; pos[:,2,1]+=1.
        return {"local_rot_mats":local,"root_positions":root,"posed_joints":pos,"global_rot_mats":local}
    class Model:
        fps=30.
        skeleton=sk
        def __call__(self,texts,counts,**kwargs):
            calls.update(texts=texts,counts=counts,**kwargs)
            n=sum(counts)
            root=np.stack([np.arange(n)/30,np.ones(n),np.zeros(n)],axis=-1).astype(np.float32)
            return {"local_rot_mats":np.tile(np.eye(3),(1,n,3,1,1)),"root_positions":root[None]}
    modules={
        "kimodo":{"load_model":lambda *a,**kw:Model()},
        "kimodo.constraints":{"FullBodyConstraintSet":Constraint,"EndEffectorConstraintSet":EE},
        "kimodo.motion_rep":{},
        "kimodo.motion_rep.feature_utils":{"compute_heading_angle":lambda pos,sk:torch.zeros(len(pos))},
        "kimodo.tools":{"seed_everything":lambda seed:None},
    }
    for name,attrs in modules.items():
        module=types.ModuleType(name); module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules,name,module)
    obj=KimodoAdapter.__new__(KimodoAdapter)
    obj.torch=torch; obj.build=lambda joints:sk; obj.complete=complete
    source=Clip(np.tile(np.eye(3),(48,3,1,1)),np.stack([100+np.arange(48)/24,np.ones(48),np.zeros(48)],axis=-1),24).validate()
    obj.source=lambda *args:(source.copy(),sk)
    return obj,calls,source


def request(op="generate"):
    return {"operation":op,"timeline":[{"text":"strut","duration":6},{"text":"wave","duration":2}],
            "context_frames":12,"transition_frames":5,"output_fps":24,"source":"test.npz","blend_frames":0}


def test_generation_uses_model_not_scene_fps(adapter):
    obj,calls,_=adapter
    result,sk,info=obj.generate(request(),lambda *a:None)
    assert calls["counts"]==[180,60]
    assert len(result.root)==192
    assert result.fps==24
    assert calls["multi_prompt"] is True
    assert calls["num_samples"]==1


def test_context_is_extra_and_original_source_exact(adapter):
    obj,calls,source=adapter
    result,sk,info=obj.generate(request("continue"),lambda *a:None)
    assert calls["counts"]==[192,60]
    assert len(calls["constraint_lst"])==2
    assert len(calls["constraint_lst"][0].frames)==12
    assert len(result.root)==48+192
    np.testing.assert_array_equal(result.root[:48],source.root)
    np.testing.assert_array_equal(result.local[:48],source.local)
    assert info["context_frames"]==12


def test_blended_continuation_count(adapter):
    obj,calls,source=adapter
    req=request("continue"); req["blend_frames"]=6
    result,_,_=obj.generate(req,lambda *a:None)
    assert len(result.root)==240
    np.testing.assert_array_equal(result.root[:40],source.root[:40])
    result.validate()


def test_invalid_transition_rejected(adapter):
    obj,_,_=adapter
    req=request(); req["transition_frames"]=61
    with pytest.raises(ValueError,match="Transition"):
        obj.generate(req,lambda *a:None)


def test_authored_constraints_enter_real_model_call(adapter,monkeypatch):
    obj,calls,source=adapter
    obj.metadata=lambda sk:{'names':['a','b','c'],'parents':[-1,0,1]}
    captured=[]
    def load(data,sk):
        captured.extend(data)
        return ['AUTHORED_CONSTRAINT']
    monkeypatch.setattr(sys.modules['kimodo.constraints'],'load_constraints_lst',load,raising=False)
    req=request('continue')
    req.update(scene_fps=24.,sequence_origin=49,
               constraints=[{'type':'root2d','start_frame':49,'end_frame':49,
                             'payload':{'smooth_root_2d':[[103.,0.]]}}])
    obj.generate(req,lambda *a:None)
    assert captured[0]['frame_indices']==[12]
    assert calls['constraint_lst'][-1]=='AUTHORED_CONSTRAINT'
    assert 0<captured[0]['smooth_root_2d'][0][0]<3


def test_fresh_root_constraint_aligns_world_origin(adapter,monkeypatch):
    obj,calls,source=adapter
    obj.metadata=lambda sk:{'names':['a','b','c'],'parents':[-1,0,1]}
    captured=[]
    def load(data,sk):
        captured.extend(data);return ['ROOT']
    monkeypatch.setattr(sys.modules['kimodo.constraints'],'load_constraints_lst',load,raising=False)
    req=request()
    req.update(scene_fps=24.,sequence_origin=1,
               constraints=[{'type':'root2d','start_frame':1,'end_frame':1,
                             'payload':{'smooth_root_2d':[[100.,20.]]}}])
    result,_,_=obj.generate(req,lambda *a:None)
    assert captured[0]['smooth_root_2d']==[[0.,0.]]
    np.testing.assert_allclose(result.root[0], [100.,1.,20.])


def test_model_is_cached_between_variations(adapter,monkeypatch):
    obj,calls,_=adapter
    original=sys.modules['kimodo'].load_model
    loaded=[]
    def load(*args,**kwargs):
        loaded.append(args);return original(*args,**kwargs)
    monkeypatch.setattr(sys.modules['kimodo'],'load_model',load)
    obj.generate(request(),lambda *a:None)
    obj.generate(request(),lambda *a:None)
    assert len(loaded)==1
