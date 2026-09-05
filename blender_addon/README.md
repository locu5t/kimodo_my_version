# Kimodo Motion Studio 0.4.0

Build: `python blender_tools/build_addon.py` from the repository root.
Install `blender_dist/kimodo_motion_studio-0.4.0.zip` in Blender 5.2 using
**Install from Disk**, and enable the add-on. A fresh install opens a local setup
dialog automatically; approve the folder once and backend preparation runs without
manual Python/venv/bridge commands. Download model weights separately in Generate.

See **BLENDER_SETUP.md** for setup, storage, retry, platform support and the
optional compiled-cleanup limitation. See **BLENDER_README.md** for the native
UI, JSON, constraints and continuation workflow inherited from 0.3.0.

Disable older duplicate Kimodo add-ons before installing. Source archives are
not Blender extension ZIPs. This is a development alpha; read the validation
report and actual CI results before assuming end-to-end GPU compatibility.
