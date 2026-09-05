import bpy


class KIMODO_UL_prompts(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.prop(item, "text", text="", emboss=False)
        row.prop(item, "duration", text="s")


class KIMODO_PT_motion_studio(bpy.types.Panel):
    bl_label = "Kimodo Motion Studio"
    bl_idname = "KIMODO_PT_motion_studio"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Kimodo'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.kimodo_motion

        box = layout.box()
        box.label(text="Prompt Timeline")
        box.template_list(
            "KIMODO_UL_prompts",
            "",
            settings,
            "prompts",
            settings,
            "prompt_index",
            rows=5,
        )
        row = box.row(align=True)
        row.operator("kimodo.add_prompt", text="Add", icon='ADD')
        row.operator("kimodo.remove_prompt", text="Remove", icon='REMOVE')
        box.operator("kimodo.import_prompts_json", icon='FILE_FOLDER')
        box.operator("kimodo.build_prompt_markers", icon='MARKER')

        box = layout.box()
        box.label(text="Continuation")
        box.prop(settings, "source_armature")
        box.prop(settings, "context_frames")
        box.prop(settings, "blend_frames")

        box = layout.box()
        box.label(text="Kimodo Backend")
        box.prop(settings, "repo_path")
        box.prop(settings, "python_path")
        box.prop(settings, "model_name")
        box.prop(settings, "output_path")
        box.label(text="Generation bridge is the next implementation stage", icon='INFO')


_CLASSES = (KIMODO_UL_prompts, KIMODO_PT_motion_studio)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
