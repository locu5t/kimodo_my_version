# SPDX-License-Identifier: GPL-3.0-or-later
"""Kimodo Motion Studio. GPU inference runs outside Blender's Python."""
bl_info = {
    "name": "Kimodo Motion Studio", "author": "locu5t and contributors",
    "version": (0, 3, 0), "blender": (4, 2, 0),
    "location": "3D View > Sidebar > Kimodo; Dope Sheet > Sidebar > Kimodo",
    "description": "Native timeline, constraint authoring, prompt JSON and motion continuation",
    "category": "Animation",
}

def register():
    from . import ui
    ui.register()

def unregister():
    from . import ui
    ui.unregister()
