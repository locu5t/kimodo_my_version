import bpy


class KimodoPromptItem(bpy.types.PropertyGroup):
    text: bpy.props.StringProperty(name="Prompt", default="A person walks with a strut")
    duration: bpy.props.FloatProperty(name="Duration (s)", default=6.0, min=0.01)
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)


class KimodoMotionSettings(bpy.types.PropertyGroup):
    prompts: bpy.props.CollectionProperty(type=KimodoPromptItem)
    prompt_index: bpy.props.IntProperty(default=0)
    source_armature: bpy.props.PointerProperty(name="Source Armature", type=bpy.types.Object)
    context_frames: bpy.props.IntProperty(name="Context Frames", default=10, min=1, max=120)
    blend_frames: bpy.props.IntProperty(name="Blend Frames", default=10, min=0, max=120)
    model_name: bpy.props.StringProperty(name="Kimodo Model", default="")
    python_path: bpy.props.StringProperty(name="Kimodo Python", subtype='FILE_PATH')
    repo_path: bpy.props.StringProperty(name="Kimodo Repo", subtype='DIR_PATH')
    output_path: bpy.props.StringProperty(name="Output", subtype='FILE_PATH')


_CLASSES = (KimodoPromptItem, KimodoMotionSettings)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.kimodo_motion = bpy.props.PointerProperty(type=KimodoMotionSettings)


def unregister():
    del bpy.types.Scene.kimodo_motion
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
