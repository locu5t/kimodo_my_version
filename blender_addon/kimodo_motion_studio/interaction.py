# SPDX-License-Identifier: GPL-3.0-or-later
"""Native GPU timeline lanes and scoped, undoable mouse interactions."""
import math
import bpy
from bpy.props import IntProperty, StringProperty, FloatProperty
from . import ui, rig
from .timeline import frame_prompts, parse_prompts, round_frame
from .authoring import TRACKS, resize_duration

_KEYS=[]
_MENU_KIND='prompt'
LEFT=116
COLORS=((.18,.48,.67,1),(.46,.33,.65,1),(.20,.54,.42,1),(.65,.40,.20,1))


def geometry(context):
    s=context.scene.kimodo_studio; region=context.region
    height=s.row_height; top=region.height-28
    # Short Timeline areas still get a prompt row; enlarge to reveal constraint lanes.
    labels=[('prompt','Text prompts')]+(list(TRACKS) if s.show_tracks else [])
    labels=labels[:max(1,min(len(labels),int((region.height-30)/height)))]
    rows=[{'kind':kind,'label':name,'y0':top-(i+1)*height,'y1':top-i*height-2}
          for i,(kind,name) in enumerate(labels)]
    hits=[]; view=region.view2d
    try:
        spans=frame_prompts(ui.enabled_prompts(context.scene),rig.scene_fps(context.scene),ui.timeline_origin(s))
        indices=[i for i,p in enumerate(s.prompts) if p.enabled]
    except ValueError:
        spans=[]; indices=[]
    for i,p in zip(indices,spans):
        row=rows[0]
        a=view.view_to_region(p.start,0,clip=False)[0]; b=view.view_to_region(p.end,0,clip=False)[0]
        if b>LEFT and a<region.width:
            hits.append(dict(row,index=i,x0=max(LEFT,a),x1=min(region.width,b),raw_x0=a,raw_x1=b,
                             start=p.start,end=p.end,label=s.prompts[i].text))
    by_kind={r['kind']:r for r in rows}
    for i,c in enumerate(s.constraints):
        kind='fullbody' if c.kind=='end-effector' else c.kind
        if kind not in by_kind:
            continue
        row=by_kind[kind]
        a=view.view_to_region(c.start_frame,0,clip=False)[0]
        b=view.view_to_region(c.end_frame+1,0,clip=False)[0]
        if c.start_frame==c.end_frame:
            a-=6; b=max(a+12,b)
        if b>LEFT and a<region.width:
            hits.append(dict(row,index=i,x0=max(LEFT,a),x1=min(region.width,b),raw_x0=a,raw_x1=b,
                             start=c.start_frame,end=c.end_frame,label=c.name))
    return rows,hits


def at(context,x,y):
    rows,hits=geometry(context)
    if x<LEFT:
        return None,None
    # Favor the most recently added item for explicit same-track overlaps.
    hit=next((h for h in reversed(hits) if h['x0']<=x<=h['x1'] and h['y0']<=y<=h['y1']),None)
    row=next((r for r in rows if r['y0']<=y<=r['y1']),None)
    return row,hit


def draw_timeline():
    context=bpy.context
    if not context.scene or not context.region or context.region.type!='WINDOW' or not hasattr(context.scene,'kimodo_studio'):
        return
    if not context.scene.kimodo_studio.overlay:
        return
    import gpu
    import blf
    from gpu_extras.batch import batch_for_shader
    s=context.scene.kimodo_studio; rows,hits=geometry(context); region=context.region
    shader=gpu.shader.from_builtin('UNIFORM_COLOR')
    def rect(x0,y0,x1,y1,color):
        if x1<=x0 or y1<=y0:
            return
        batch=batch_for_shader(shader,'TRIS',{'pos':[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]},indices=[(0,1,2),(0,2,3)])
        shader.bind(); shader.uniform_float('color',color); batch.draw(shader)
    def text(x,y,label,width,color=(.95,.95,.95,1)):
        blf.size(0,11)
        while label and blf.dimensions(0,label)[0]>width:
            label=label[:-1]
        if label:
            blf.position(0,x,y,0); blf.color(0,*color); blf.draw(0,label)
    gpu.state.blend_set('ALPHA')
    try:
        for row in rows:
            rect(0,row['y0'],region.width,row['y1'],(.10,.115,.135,.97))
            rect(0,row['y0'],LEFT,row['y1'],(.16,.18,.21,1))
            text(7,row['y0']+7,row['label'],LEFT-14)
        for hit in hits:
            prompt=hit['kind']=='prompt'; i=hit['index']
            selected=(s.index if prompt else s.constraint_index)==i
            enabled=True if prompt else s.constraints[i].enabled
            color=COLORS[i%len(COLORS)] if prompt else (.58,.39,.22,1)
            if not enabled:
                color=(.27,.28,.30,1)
            x0,x1,y0,y1=hit['x0'],hit['x1'],hit['y0']+2,hit['y1']-2
            if selected:
                rect(x0,y0,x1,y1,(.85,.85,.85,1)); x0+=1;y0+=1;x1-=1;y1-=1
            rect(x0,y0,x1,y1,color)
            if x1-x0>26:
                text(x0+5,y0+5,hit['label'],x1-x0-13)
                rect(x1-4,y0+4,x1-2,y1-4,(.87,.87,.9,.7))
        x=region.view2d.view_to_region(context.scene.frame_current,0,clip=False)[0]
        if LEFT<x<region.width and rows:
            rect(x-1,rows[-1]['y0'],x+1,rows[0]['y1'],(.95,.30,.18,1))
    finally:
        gpu.state.blend_set('NONE')


class KMD_OT_PromptDialog(bpy.types.Operator):
    bl_idname='kimodo.prompt_dialog'; bl_label='Edit motion prompt'
    bl_options={'REGISTER','UNDO'}
    index:IntProperty(default=0)
    text:StringProperty(name='Motion description')
    duration:FloatProperty(name='Duration (seconds)',min=.001,max=1200.)
    def invoke(self,context,event):
        s=context.scene.kimodo_studio
        if not 0<=self.index<len(s.prompts):
            return {'CANCELLED'}
        p=s.prompts[self.index]; self.text=p.text; self.duration=p.duration
        return context.window_manager.invoke_props_dialog(self,width=520)
    def draw(self,context):
        self.layout.prop(self,'text');self.layout.prop(self,'duration')
    def execute(self,context):
        s=context.scene.kimodo_studio
        try:
            candidate=[{'text':p.text,'duration':p.duration} for p in s.prompts]
            candidate[self.index]={'text':self.text,'duration':self.duration}
            frame_prompts(parse_prompts(candidate),rig.scene_fps(context.scene))
            s.prompts[self.index].text=self.text; s.prompts[self.index].duration=self.duration
        except (ValueError,IndexError) as exc:
            self.report({'ERROR'},str(exc)); return {'CANCELLED'}
        return {'FINISHED'}


class KMD_MT_Timeline(bpy.types.Menu):
    bl_idname='KMD_MT_timeline'; bl_label='Kimodo timeline'
    def draw(self,context):
        s=context.scene.kimodo_studio
        if _MENU_KIND=='prompt':
            self.layout.operator('kimodo.prompt_dialog',text='Edit prompt').index=s.index
            self.layout.operator('kimodo.edit_prompt',text='Delete prompt').action='REMOVE'
        else:
            self.layout.operator('kimodo.constraint_action',text='Edit pose / waypoint').action='EDIT'
            self.layout.operator('kimodo.constraint_action',text='Recapture pose').action='RECAPTURE'
            self.layout.operator('kimodo.constraint_action',text='Delete constraint').action='DELETE'


class KMD_OT_Interact(bpy.types.Operator):
    bl_idname='kimodo.timeline_interact'; bl_label='Edit Kimodo timeline'
    bl_options={'REGISTER','UNDO','BLOCKING'}
    @classmethod
    def poll(cls,context):
        return (context.area is not None and context.area.type=='DOPESHEET_EDITOR' and context.region is not None
                and context.region.type=='WINDOW' and context.scene is not None
                and hasattr(context.scene,'kimodo_studio') and context.scene.kimodo_studio.overlay
                and context.scene.kimodo_studio.editing_index<0 and ui._ACTIVE is None)
    def invoke(self,context,event):
        global _MENU_KIND
        if event.alt or event.oskey:
            return {'PASS_THROUGH'}
        row,hit=at(context,event.mouse_region_x,event.mouse_region_y)
        if not row:
            return {'PASS_THROUGH'}
        s=context.scene.kimodo_studio
        self._scene=context.scene; self._area=context.area; self._region=context.region
        self._x=event.mouse_region_x; self._frame=context.region.view2d.region_to_view(event.mouse_region_x,event.mouse_region_y)[0]
        self._kind=row['kind']; self._moved=False; self._mode='CAPTURE'; self._index=-1
        if hit:
            self._index=hit['index']; self._start=hit['start']; self._end=hit['end']
            if self._kind=='prompt':
                s.index=self._index; self._duration=s.prompts[self._index].duration
                _MENU_KIND='prompt'
            else:
                s.constraint_index=self._index; _MENU_KIND='constraint'
            if event.type=='RIGHTMOUSE':
                bpy.ops.wm.call_menu(name='KMD_MT_timeline'); return {'FINISHED'}
            if event.value=='DOUBLE_CLICK' and self._kind=='prompt':
                bpy.ops.kimodo.prompt_dialog('INVOKE_DEFAULT',index=self._index); return {'FINISHED'}
            self._mode='MOVE'
            if abs(event.mouse_region_x-hit['raw_x1'])<=8:
                self._mode='RIGHT'
            elif self._kind!='prompt' and self._start!=self._end and abs(event.mouse_region_x-hit['raw_x0'])<=8:
                self._mode='LEFT'
            if self._kind!='prompt' and self._start==self._end and not event.ctrl:
                self._mode='MOVE'
        elif event.type=='RIGHTMOUSE':
            return {'PASS_THROUGH'}
        elif row['kind']=='prompt':
            bpy.ops.kimodo.edit_prompt(action='ADD'); return {'FINISHED'}
        elif not event.ctrl:
            frame=max(1,round_frame(self._frame))
            bpy.ops.kimodo.capture_constraint(kind=self._kind,start_frame=frame,end_frame=frame)
            return {'FINISHED'}
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    def restore(self):
        s=self._scene.kimodo_studio
        if self._index>=0:
            if self._kind=='prompt' and self._mode=='RIGHT':
                s.prompts[self._index].duration=self._duration
            elif self._kind!='prompt':
                item=s.constraints[self._index]; item.start_frame=self._start; item.end_frame=self._end
    def modal(self,context,event):
        if context.scene!=self._scene or context.area!=self._area:
            self.restore();return {'CANCELLED'}
        s=self._scene.kimodo_studio
        if event.type in {'ESC','RIGHTMOUSE'} and event.value=='PRESS':
            self.restore();ui.redraw();return {'CANCELLED'}
        if event.type=='MOUSEMOVE':
            f=self._region.view2d.region_to_view(event.mouse_region_x,event.mouse_region_y)[0]
            self._moved=self._moved or abs(event.mouse_region_x-self._x)>4
            if self._index>=0 and self._moved:
                if self._kind=='prompt' and self._mode=='RIGHT':
                    other=sum(p.duration for i,p in enumerate(s.prompts) if p.enabled and i!=self._index)
                    duration=resize_duration(self._start,f,rig.scene_fps(self._scene))
                    s.prompts[self._index].duration=min(max(.001,1200.-other),duration)
                elif self._kind!='prompt':
                    item=s.constraints[self._index]; delta=round_frame(f-self._frame)
                    if self._mode=='MOVE':
                        delta=max(delta,1-self._start)
                        item.start_frame=self._start+delta;item.end_frame=self._end+delta
                    elif self._mode=='RIGHT':
                        item.end_frame=max(self._start,round_frame(f)-1)
                    elif self._mode=='LEFT':
                        item.start_frame=min(self._end,max(1,round_frame(f)))
                ui.redraw()
        if event.type=='LEFTMOUSE' and event.value=='RELEASE':
            f=self._region.view2d.region_to_view(event.mouse_region_x,event.mouse_region_y)[0]
            if self._mode=='CAPTURE':
                a,b=sorted((max(1,round_frame(self._frame)),max(1,round_frame(f))))
                bpy.ops.kimodo.capture_constraint(kind=self._kind,start_frame=a,end_frame=b)
            elif self._kind=='prompt' and self._mode=='MOVE' and self._moved:
                try:
                    spans=frame_prompts(ui.enabled_prompts(self._scene),rig.scene_fps(self._scene),ui.timeline_origin(s))
                    ids=[i for i,p in enumerate(s.prompts) if p.enabled]
                    target=ids[-1]
                    for i,span in zip(ids,spans):
                        if f<(span.start+span.end)/2:
                            target=i;break
                    s.prompts.move(self._index,target);s.index=target
                except ValueError:
                    self.restore();return {'CANCELLED'}
            ui.update_markers(self._scene)
            return {'FINISHED'}
        return {'RUNNING_MODAL'}


CLASSES=(KMD_OT_PromptDialog,KMD_MT_Timeline,KMD_OT_Interact)

def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    config=bpy.context.window_manager.keyconfigs.addon
    if config:
        keymap=config.keymaps.new(name='Dopesheet',space_type='DOPESHEET_EDITOR')
        for button,value in [('LEFTMOUSE','PRESS'),('LEFTMOUSE','DOUBLE_CLICK'),('RIGHTMOUSE','PRESS')]:
            item=keymap.keymap_items.new('kimodo.timeline_interact',button,value,any=True,head=True)
            _KEYS.append((keymap,item))

def unregister():
    for keymap,item in _KEYS:
        keymap.keymap_items.remove(item)
    _KEYS.clear()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
