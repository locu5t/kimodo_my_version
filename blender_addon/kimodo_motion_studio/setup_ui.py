# SPDX-License-Identifier: GPL-3.0-or-later
"""First-enable setup dialog and Blender-main-thread installer monitoring."""
import atexit
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from . import managed_setup as core

_JOB = None
_ORIGINAL = {}
_REGISTERED = False
_ATTEMPTS = 0


def prefs():
    from . import ui
    return ui.preferences(bpy.context)


def save_preferences():
    try:
        bpy.ops.wm.save_userpref()
    except RuntimeError:
        pass


def setup_paths(p):
    return core.paths(bpy.path.abspath(p.setup_root) if p.setup_root else core.default_root(), p.setup_compute)


def use_managed(p):
    paths = setup_paths(p)
    if not core.ready(paths['root'], p.setup_compute):
        raise ValueError('Local backend is not ready. Use Set up local backend / Retry first')
    p.backend = 'NATIVE'
    p.python_path, p.repo_path = str(paths['python']), str(paths['source'])
    if not p.models_root:
        p.models_root = str(paths['root'] / 'models')
    if not p.output_root:
        p.output_root = str(paths['root'] / 'outputs')
    return paths


def draw_setup(layout, context, compact=False):
    p = prefs()
    box = layout.box()
    box.label(text='Local backend setup', icon='PREFERENCES')
    box.prop(p, 'setup_mode', text='Backend')
    if p.setup_mode == 'EXTERNAL':
        box.label(text='Using your existing Python / repository settings')
        return
    if not core.supported():
        box.label(text='Managed setup: Windows x64 / Linux x86-64 only', icon='ERROR')
        return
    paths = setup_paths(p)
    is_ready = core.ready(paths['root'], p.setup_compute)
    status = core.read_json(paths['status'])
    message = ('Backend ready; model weights are configured separately' if is_ready else
               status.get('message', 'Install once; subsequent launches are automatic'))
    for i in range(0, min(len(message), 320), 48):
        box.label(text=message[i:i+48], icon='CHECKMARK' if is_ready and i == 0 else 'NONE')
    if _JOB:
        box.operator('kimodo.cancel_setup', text='Cancel setup', icon='CANCEL')
    else:
        row = box.row()
        row.enabled = bool(getattr(bpy.app, 'online_access', True))
        row.operator('kimodo.setup_backend', text='Verify / repair backend' if is_ready else 'Set up local backend / Retry', icon='IMPORT')
        if not getattr(bpy.app, 'online_access', True):
            box.label(text='Enable Allow Online Access in Blender preferences', icon='INFO')
    box.operator('kimodo.open_setup_folder', text='Open setup folder / logs', icon='FILE_FOLDER')
    box.prop(p, 'setup_details', text='Details / advanced')
    if p.setup_details:
        box.label(text=str(paths['home']))
        box.label(text='Private Python 3.11; Blender Python stays unchanged')
        box.label(text='No admin, Git or manual bridge startup required')
        box.label(text='Managed profile excludes optional C++ foot cleanup')
        box.label(text='Models: use Generate > Local model storage')


def extended_draw(self, context):
    draw_setup(self.layout, context)
    if self.setup_mode == 'EXTERNAL' or self.setup_details:
        _ORIGINAL['draw'](self, context)


def guarded_invoke(self, context, event):
    if _JOB:
        self.report({'ERROR'}, 'Backend setup is running; finish or cancel it before launching a job')
        return {'CANCELLED'}
    p = prefs()
    if p.setup_mode == 'MANAGED':
        try:
            use_managed(p)
            if self.operation in ('generate', 'continue') and context.scene.kimodo_studio.postprocess:
                raise ValueError('Managed setup excludes optional C++ MotionCorrection. Disable foot-skate cleanup or select an external backend with it installed')
            if p.setup_compute == 'cpu':
                context.scene.kimodo_studio.device = 'cpu'
        except (ValueError, OSError) as exc:
            self.report({'ERROR'}, str(exc)); return {'CANCELLED'}
    return _ORIGINAL['invoke'](self, context, event)


def extend_preferences(ui):
    """Compose preferences/operators BEFORE RNA registration, not monkeypatch RNA."""
    if _ORIGINAL:
        return
    _ORIGINAL.update(draw=ui.KMD_Preferences.draw, invoke=ui.KMD_OT_Run.invoke)
    ui.KMD_Preferences.__annotations__.update({
        'setup_mode': EnumProperty(name='Backend management', items=[
            ('MANAGED', 'Automatic local setup', 'The add-on manages its own isolated environment'),
            ('EXTERNAL', 'Existing environment (advanced)', 'Keep your manually configured native/WSL backend')], default='MANAGED'),
        'setup_root': StringProperty(name='Backend install folder', subtype='DIR_PATH'),
        'setup_compute': EnumProperty(name='Compute', items=[('cu128', 'NVIDIA GPU (CUDA 12.8)', 'Requires a compatible NVIDIA GPU and driver'),
                                                            ('cpu', 'CPU only', 'No GPU acceleration; generation can be very slow')], default='cu128'),
        'setup_prompted': BoolProperty(default=False),
        'setup_details': BoolProperty(name='Advanced backend settings', default=False),
    })
    ui.KMD_Preferences.draw = extended_draw
    ui.KMD_OT_Run.invoke = guarded_invoke


def bootstrap_command(script, root, compute):
    """Use Blender's bundled executable; never require a system Python install."""
    args = ['--root', str(root), '--compute', compute, '--approved', '--parent-pid', str(os.getpid())]
    names = ['python.exe'] if os.name == 'nt' else [f'python{sys.version_info.major}.{sys.version_info.minor}', 'python3']
    candidates = [Path(sys.executable)] if Path(sys.executable).name.lower().startswith('python') else []
    for prefix in (Path(sys.prefix), Path(sys.base_prefix)):
        candidates += [prefix / 'bin' / name for name in names]
    for executable in candidates:
        if executable.is_file():
            return [str(executable), '-I', str(script), *args]
    if bpy.app.binary_path and Path(bpy.app.binary_path).is_file():
        return [bpy.app.binary_path, '--background', '--factory-startup', '--disable-autoexec',
                '--python-exit-code', '1', '--python', str(script), '--', *args]
    raise ValueError('Could not locate Blender bundled Python or Blender executable')


class KMD_OT_Setup(bpy.types.Operator):
    bl_idname = 'kimodo.setup_backend'
    bl_label = 'Set up local Kimodo'
    bl_description = 'Download and configure an isolated local backend; does not download model weights'
    folder: StringProperty(name='Install folder', subtype='DIR_PATH')
    compute: EnumProperty(name='Compute', items=[('cu128', 'NVIDIA GPU (CUDA 12.8)', ''), ('cpu', 'CPU only', '')], default='cu128')
    _approved = False

    def invoke(self, context, event):
        p = prefs()
        self.folder = p.setup_root or str(core.default_root())
        self.compute = p.setup_compute
        self._approved = False
        return context.window_manager.invoke_props_dialog(self, width=580)

    def draw(self, context):
        layout = self.layout
        layout.label(text='One-time local setup for Blender 5.2', icon='INFO')
        layout.prop(self, 'folder'); layout.prop(self, 'compute')
        layout.label(text='Downloads Python, PyTorch and Kimodo dependencies (several GB).')
        layout.label(text='No changes to Blender Python, system Python, PATH or drivers.')
        layout.label(text='Your model folders and existing environments are not deleted.')
        layout.label(text='Model weights / gated Hugging Face access are a separate step.')
        layout.label(text='Optional C++ foot-skate cleanup is not included in this profile.')
        layout.label(text='Click OK to approve downloads and begin automatic setup.')
        if not getattr(bpy.app, 'online_access', True):
            layout.label(text='Allow Online Access must be enabled in Blender preferences.', icon='ERROR')
        # Only a displayed interactive confirmation may approve installation.
        self._approved = True

    def execute(self, context):
        global _JOB
        from . import ui
        if not self._approved or bpy.app.background:
            self.report({'ERROR'}, 'Open the setup dialog and approve its downloads first')
            return {'CANCELLED'}
        if _JOB or ui._ACTIVE:
            self.report({'ERROR'}, 'Finish or cancel the current Kimodo job first'); return {'CANCELLED'}
        if not getattr(bpy.app, 'online_access', True):
            self.report({'ERROR'}, 'Enable Allow Online Access in Blender preferences'); return {'CANCELLED'}
        logfile = None
        try:
            if not core.supported():
                raise ValueError('Managed setup supports Windows x64 and Linux x86-64')
            folder = Path(bpy.path.abspath(self.folder)).expanduser()
            if not self.folder.strip() or not folder.is_absolute():
                raise ValueError('Choose an absolute local installation folder')
            if os.name == 'nt' and str(folder).startswith('\\\\'):
                raise ValueError('Choose a local drive, not a network/UNC folder')
            paths = core.paths(folder, self.compute)
            core.own_runtime(paths)
            with core.exclusive_lock(paths['lock']):
                paths['cancel'].unlink(missing_ok=True)
            command = bootstrap_command(Path(__file__).with_name('managed_setup.py'), paths['root'], self.compute)
            logpath = paths['home'] / ('setup-' + uuid.uuid4().hex[:8] + '.log')
            logfile = logpath.open('w', encoding='utf-8')
            kwargs = dict(stdout=logfile, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                          shell=False, env=core.clean_environment(paths, self.compute))
            if os.name == 'nt':
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
            proc = subprocess.Popen(command, **kwargs)
            p = prefs()
            p.setup_root = str(paths['root']); p.setup_compute = self.compute
            p.setup_mode = 'MANAGED'; p.setup_prompted = True
            _JOB = dict(process=proc, log=logfile, paths=paths, logfile=logpath, started=time.monotonic())
            save_preferences()
            if not bpy.app.timers.is_registered(poll_setup):
                bpy.app.timers.register(poll_setup, first_interval=.3, persistent=True)
            self.report({'INFO'}, 'Local backend setup started. Progress is shown in Kimodo > Local backend setup')
            return {'FINISHED'}
        except Exception as exc:
            if logfile:
                logfile.close()
            self.report({'ERROR'}, str(exc)); return {'CANCELLED'}


class KMD_OT_CancelSetup(bpy.types.Operator):
    bl_idname = 'kimodo.cancel_setup'
    bl_label = 'Cancel local setup'
    def execute(self, context):
        cancel_setup()
        return {'FINISHED'}


def cancel_setup():
    if _JOB:
        try:
            _JOB['paths']['cancel'].write_text('cancel', encoding='utf-8')
        except OSError:
            pass


def poll_setup():
    global _JOB
    if not _JOB or not _REGISTERED:
        return None
    from . import ui
    job = _JOB
    if not getattr(bpy.app, 'online_access', True):
        cancel_setup()
    ui.redraw()
    if job['process'].poll() is None:
        return .3
    job['log'].close()
    p = prefs()
    try:
        if job['process'].returncode == 0 and core.ready(job['paths']['root'], p.setup_compute):
            use_managed(p)
            if bpy.context.scene and p.setup_compute == 'cpu':
                bpy.context.scene.kimodo_studio.device = 'cpu'
            save_preferences()
        elif core.read_json(job['paths']['status']).get('state') not in ('error', 'cancelled'):
            core.atomic_json(job['paths']['status'], {'state': 'error', 'message': f'Setup exited with {job["process"].returncode}. Read {job["logfile"].name}'})
    except Exception as exc:
        print('Kimodo setup completion:', exc)
    finally:
        _JOB = None
    return None


class KMD_OT_OpenSetup(bpy.types.Operator):
    bl_idname = 'kimodo.open_setup_folder'
    bl_label = 'Open local setup folder'
    def execute(self, context):
        p = setup_paths(prefs())
        if p['home'].is_dir():
            bpy.ops.wm.path_open(filepath=str(p['home'])); return {'FINISHED'}
        self.report({'INFO'}, 'No setup folder has been created yet')
        return {'CANCELLED'}


class KMD_PT_Setup(bpy.types.Panel):
    bl_label = 'Kimodo Local Backend'
    bl_idname = 'KMD_PT_setup'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Kimodo'
    bl_order = -20
    def draw(self, context):
        draw_setup(self.layout, context, True)


def first_enable():
    global _ATTEMPTS
    if not _REGISTERED or bpy.app.background:
        return None
    _ATTEMPTS += 1
    try:
        p = prefs()
        # Preserve a configured 0.3.x backend when migrating existing users.
        if not p.is_property_set('setup_mode') and (p.python_path or p.repo_path or p.backend == 'WSL'):
            p.setup_mode = 'EXTERNAL'; p.setup_prompted = True
        if p.setup_mode != 'MANAGED':
            return None
        paths = setup_paths(p)
        if core.ready(paths['root'], p.setup_compute):
            use_managed(p); return None
        if p.setup_prompted or not core.supported():
            return None
        if not bpy.context.window:
            return .5 if _ATTEMPTS < 30 else None
        p.setup_prompted = True
        save_preferences()
        bpy.ops.kimodo.setup_backend('INVOKE_DEFAULT')
    except (KeyError, AttributeError, RuntimeError):
        return .5 if _ATTEMPTS < 30 else None
    return None


CLASSES = (KMD_OT_Setup, KMD_OT_CancelSetup, KMD_OT_OpenSetup, KMD_PT_Setup)


def register():
    global _REGISTERED, _ATTEMPTS
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    _REGISTERED, _ATTEMPTS = True, 0
    atexit.register(cancel_setup)
    if not bpy.app.background:
        bpy.app.timers.register(first_enable, first_interval=1.)


def unregister():
    global _REGISTERED, _JOB
    _REGISTERED = False
    cancel_setup()  # Worker observes this and stops its child process tree.
    if _JOB:
        _JOB['log'].close(); _JOB = None
    atexit.unregister(cancel_setup)
    for timer in (first_enable, poll_setup):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
