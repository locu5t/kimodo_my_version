"""Real bpy smoke test: Blender --background --factory-startup --python THIS_FILE.

Also runs as `python blender_tools/blender_smoke_test.py` with the official bpy
wheel. No Kimodo weights, torch, GPU inference, or interactive graphics required.
"""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
import bpy
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'blender_addon'))
import kimodo_motion_studio
from kimodo_motion_studio import rig
from kimodo_motion_studio.motion_math import Clip, save_portable, load_portable
from kimodo_motion_studio import studio, ui
kimodo_motion_studio.register()
checks=[]
try:
    scene=bpy.context.scene
    scene.render.fps=30; scene.render.fps_base=1.
    local=np.tile(np.eye(3),(4,3,1,1))
    roots=np.array([[0,1,0],[.1,1,0],[.2,1,0],[.3,1,0]],dtype=np.float32)
    rest=np.array([[0,0,0],[0,.5,0],[0,1,0]])
    meta={'name':'test_only','names':['root','mid','tip'],'parents':[-1,0,1],
          'root_idx':0,'rest_joints':rest.tolist(),
          'rest_global_rot_mats':np.tile(np.eye(3),(3,1,1)).tolist(),'up_axis':'Y','unit':'meter'}
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'fixture.blender.npz'
        save_portable(path,Clip(local,roots,30),meta,
                      {'posed_joints':roots[:,None,:]+rest[None,:,:], 'global_rot_mats':local})
        obj=None
        for done,total,obj in rig.bake_motion(path,scene,1,'Smoke Test'):
            pass
        assert obj is not None and obj.animation_data.action is not None
        assert len(rig.action_curves(obj.animation_data.action))>0
        scene.frame_set(3)
        assert np.allclose(obj.pose.bones['root'].matrix.translation,[.2,0,1],atol=1e-5)
        out=Path(tmp)/'roundtrip.blender.npz'
        rig.snapshot(obj,scene,1,4,out)
        recovered,_,_=load_portable(out)
        np.testing.assert_allclose(recovered.root,roots,atol=1e-5)
        np.testing.assert_allclose(recovered.local,local,atol=1e-5)
        checks.append('registration, new Action, FK axes and rig roundtrip')
        before=set(bpy.data.objects)
        pending=rig.bake_motion(path,scene,10,'Rollback Test')
        next(pending); pending.close()
        assert set(bpy.data.objects)==before
        checks.append('cancelled bake rolls back its new objects')
        s=scene.kimodo_studio;s.rig_source=obj
        prompt_path=Path(tmp)/'prompts.json'
        prompt_path.write_text('[{"text":"walk with a strut","duration":6},{"text":"wave","duration":2}]')
        assert bpy.ops.kimodo.import_prompts(filepath=str(prompt_path))=={'FINISHED'}
        assert len(s.prompts)==2
        assert len([m for m in scene.timeline_markers if m.name.startswith('KMD::')])==3
        checks.append('timed JSON import and native timeline markers')
        scene.frame_set(2)
        assert bpy.ops.kimodo.capture_constraint(use_playhead=True)=={'FINISHED'}
        assert len(s.constraints)==1 and s.constraints[0].start_frame==2
        source_action=obj.animation_data.action
        assert bpy.ops.kimodo.constraint_action(action='EDIT')=={'FINISHED'}
        assert s.edit_object is not None and s.edit_object!=obj
        assert bpy.ops.kimodo.constraint_action(action='APPLY')=={'FINISHED'}
        assert s.edit_object is None and obj.animation_data.action==source_action
        checks.append('capture, edit and apply constraint without modifying source Action')
        data=studio.make_project(scene)
        studio.apply_project(scene,data)
        assert len(s.constraints)==1 and len(s.prompts)==2
        checks.append('project JSON roundtrip')
        for section in ('GENERATE','CONSTRAINTS','FILES','VISUALIZE','HELP'):
            s.tab=section
        # Native UI property registration is checked; mouse/GPU draw validation is interactive.
        s.model_preset='Kimodo-G1-RP-v1'
        assert not s.custom_model and s.skeleton_choice=='G1' and s.dataset=='RP' and s.version_choice=='v1'
        s.model_preset='Kimodo-SOMA-RP-v1.1'
        assert s.model=='Kimodo-SOMA-RP-v1.1'
        assert hasattr(bpy.ops.kimodo, 'download_assets')
        props=ui.KMD_Preferences.__annotations__
        assert all(k in props for k in ('models_root','manual_model_path','model_cache_root','model_mode'))
        checks.append('model preset mapping, storage preferences and explicit-download operator registered')
        checks.append('all native panel sections registered')
    report={'blender_version':bpy.app.version_string,'status':'passed','checks':checks,
            'interactive_mouse_and_gpu_draw':'not tested','real_kimodo_inference':'not tested'}
    (root/'blender-smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
finally:
    kimodo_motion_studio.unregister()
