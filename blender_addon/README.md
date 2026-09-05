# Native Kimodo Motion Studio for Blender

The active implementation is `kimodo_motion_studio/` (version 0.3.0).
It consolidates the earlier prompt-only scaffold and local generation prototype
into one extension: native authoring panels, interactive prompt/constraint lanes,
timed JSON import/export, pose/path constraints, motion import and continuation.

See [the complete guide](../BLENDER_README.md) for installation, controls, validation
status and known limits. Build the installable ZIP with:

```shell
python blender_tools/build_addon.py
```

Do not install the repository source ZIP as a Blender extension. Disable the old
`kimodo_blender` alpha before installing this version. The original Kimodo Python
model and web demo are unchanged.

See `BLENDER_README.md` for local Download folder / Manual path storage and explicit offline generation.
