bl_info = {
    "name": "Kimodo Motion Studio",
    "author": "locu5t / OpenAI",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Kimodo",
    "description": "Blender integration for Kimodo prompt timelines, motion continuation, and blending",
    "category": "Animation",
}

from . import operators, panels, properties

_modules = (properties, operators, panels)


def register():
    for module in _modules:
        module.register()


def unregister():
    for module in reversed(_modules):
        module.unregister()
