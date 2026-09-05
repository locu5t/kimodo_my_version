# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated, explicitly approved backend installation. Standard library only.

This module is also a standalone process entry point. It never installs into the
interpreter running it. All package operations name the owned backend venv.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import time
import traceback
from urllib.request import Request, urlopen
import uuid
import zipfile

PROFILE = 'py311-torch271-r1'
PYTHON_VERSION = '3.11.13'
TORCH_VERSION = '2.7.1'
SOURCE_COMMIT = 'acef5d03b4a2ae9420f44972d751e02acb8e0bbd'
SOURCE_TREE = '60c8987f5cb11f45de641f0dc50ee8a7421baba8'
SOURCE_URL = f'https://codeload.github.com/locu5t/kimodo_my_version/zip/{SOURCE_COMMIT}'
# Version-specific PyPI artifacts, with published SHA-256 digests (not latest).
UV_ASSETS = {
    'Windows': (
        'https://files.pythonhosted.org/packages/38/82/94f08992eeb193dc3d5baac437d1867cd37f040f34c7b1a4b1bde2bc4b4b/uv-0.8.22-py3-none-win_amd64.whl',
        'cda349c9ea53644d8d9ceae30db71616b733eb5330375ab4259765aef494b74e'),
    'Linux': (
        'https://files.pythonhosted.org/packages/3a/a9/a83cee9b8cf63e57ce64ba27c77777cc66410e144fd178368f55af1fa18d/uv-0.8.22-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
        '8efec4ef5acddc35f0867998c44e0b15fc4dace1e4c26d01443871a2fbb04bf6'),
}


class Cancelled(RuntimeError):
    pass


class SetupBusy(RuntimeError):
    pass


def read_json(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {} if fallback is None else fallback


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def supported(system=None, machine=None):
    system, machine = system or platform.system(), machine or platform.machine()
    return system in UV_ASSETS and machine.lower() in ('amd64', 'x86_64')


def default_root():
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData/Local'))) / 'KimodoMotionStudio'
    return Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share'))) / 'kimodo-motion-studio'


def paths(root, compute, system=None):
    if compute not in ('cu128', 'cpu'):
        raise ValueError('Choose NVIDIA CUDA 12.8 or CPU')
    system = system or platform.system()
    if system not in UV_ASSETS:
        raise ValueError('Managed setup supports Windows x64 and Linux x86-64; use an external backend on other systems')
    root = Path(root).expanduser().resolve()
    home = root / 'runtimes' / f'{PROFILE}-{system.lower()}-{compute}'
    return {'root': root, 'home': home, 'source': home / 'source',
            'venv': home / 'venv', 'python': home / 'venv' / ('Scripts/python.exe' if system == 'Windows' else 'bin/python'),
            'ready': home / 'ready.json', 'status': home / 'status.json',
            'log': home / 'setup.log', 'cancel': home / 'cancel.request',
            'owner': home / 'owner.json', 'lock': home / 'setup.lock'}


def ready(root, compute):
    """A completion marker alone is insufficient: also check identity and files."""
    try:
        p = paths(root, compute)
        data = read_json(p['ready'])
        return (data.get('profile') == PROFILE and data.get('source_commit') == SOURCE_COMMIT
                and data.get('compute') == compute and data.get('verified') is True
                and data.get('home') == str(p['home']) and p['python'].is_file()
                and (p['source'] / 'kimodo/__init__.py').is_file())
    except (OSError, ValueError):
        return False


def own_runtime(p):
    """Refuse to install over another environment or follow a runtime symlink."""
    if p['home'].is_symlink():
        raise ValueError('Managed runtime must not be a symlink')
    if p['home'].exists() and any(p['home'].iterdir()):
        owner = read_json(p['owner'])
        if owner.get('application') != 'KimodoMotionStudio' or owner.get('profile') != PROFILE:
            raise ValueError('Destination contains unowned files. Choose a different installation folder')
    p['home'].mkdir(parents=True, exist_ok=True)
    if p['home'].resolve() != p['home'] or not p['home'].is_relative_to(p['root']):
        raise ValueError('Runtime resolves outside its selected installation folder')
    for key in ('source', 'venv', 'ready', 'owner', 'lock', 'log', 'status', 'cancel'):
        if p[key].is_symlink():
            raise ValueError(f'Managed {key} must not be a symlink')
    atomic_json(p['owner'], {'application': 'KimodoMotionStudio', 'profile': PROFILE})


@contextmanager
def exclusive_lock(path):
    """OS locks release on a crash; never delete or steal another process's lock."""
    f = Path(path).open('a+b')
    try:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SetupBusy('Another setup is using this runtime. Close it before retrying') from exc
        yield
    finally:
        f.close()


def clean_environment(p, compute):
    # Do not inherit another activated venv, pip/uv target, or Blender's Python paths.
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'UV_', 'PIP_', 'CONDA'))
           and k not in ('VIRTUAL_ENV', '_CE_CONDA', '_CE_M', 'LD_LIBRARY_PATH', 'LD_PRELOAD')}
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               UV_CACHE_DIR=str(p['home'] / 'cache/uv'),
               UV_PYTHON_INSTALL_DIR=str(p['home'] / 'python'),
               UV_PYTHON_INSTALL_BIN='false', UV_PYTHON_INSTALL_REGISTRY='false',
               UV_NO_MODIFY_PATH='1', UV_LINK_MODE='copy', UV_NO_PROGRESS='1',
               SKIP_MOTION_CORRECTION_IN_SETUP='1')
    return env


def parent_alive(pid):
    if os.name != 'nt':
        return os.getppid() == pid
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: still alive
    finally:
        kernel.CloseHandle(handle)


def stop_child(proc):
    if proc.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill.exe', '/PID', str(proc.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            import signal
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.wait(timeout=5)


def git_object(kind, content):
    return hashlib.sha1(f'{kind} {len(content)}\0'.encode() + content).digest()


def git_tree(files):
    """Hash path -> (mode, blob digest), exactly as Git does, without requiring Git."""
    tree = {}
    for path, value in files.items():
        parts = path.split('/')
        node = tree
        for name in parts[:-1]:
            node = node.setdefault(name, {})
            if not isinstance(node, dict):
                raise ValueError('File/directory collision')
        if parts[-1] in node:
            raise ValueError('Duplicate archive path')
        node[parts[-1]] = value
    def digest(node):
        result = bytearray()
        for name, value in sorted(node.items(), key=lambda kv: (kv[0] + ('/' if isinstance(kv[1], dict) else '')).encode()):
            mode, blob = ('40000', digest(value)) if isinstance(value, dict) else value
            result.extend(f'{mode} {name}\0'.encode()); result.extend(blob)
        return git_object('tree', bytes(result))
    return digest(tree).hex()


def source_members(z):
    members, files, prefix, seen = [], {}, None, set()
    total = 0
    for item in z.infolist():
        raw = item.filename
        parts = PurePosixPath(raw).parts
        if (not parts or raw.startswith('/') or '\\' in raw or ':' in raw or '..' in parts
                or any(x in ('.git', '') for x in parts)):
            raise ValueError('Unsafe source archive path')
        if prefix is None:
            prefix = parts[0]
        if parts[0] != prefix:
            raise ValueError('Source archive must contain one root directory')
        if item.is_dir():
            continue
        if len(parts) < 2:
            raise ValueError('Source file lacks repository root')
        name = '/'.join(parts[1:])
        folded = name.casefold()
        if folded in seen:
            raise ValueError('Duplicate/case-colliding source path')
        seen.add(folded)
        bits = item.external_attr >> 16
        if stat.S_ISLNK(bits) or (stat.S_IFMT(bits) not in (0, stat.S_IFREG)):
            raise ValueError('Links/devices are not allowed in source archives')
        total += item.file_size
        if total > 1024**3 or len(members) > 20000:
            raise ValueError('Source archive exceeds safety limits')
        mode = '100755' if bits & 0o111 else '100644'
        files[name] = (mode, git_object('blob', z.read(item)))
        members.append((item, name))
    return members, files


def install_source(archive, destination, expected_tree=SOURCE_TREE):
    """Verify the entire pinned Git tree BEFORE extracting or running setup.py."""
    destination = Path(destination)
    with zipfile.ZipFile(archive) as z:
        members, files = source_members(z)
        if git_tree(files) != expected_tree:
            raise ValueError('Kimodo archive does not match the pinned Git tree; nothing was installed')
        if destination.exists():
            # Do not silently overwrite user edits or an interrupted, unverified tree.
            for _, name in members:
                file = destination / name
                if file.is_symlink() or not file.is_file() or git_object('blob', file.read_bytes()) != files[name][1]:
                    raise ValueError('Managed source is modified/incomplete. Choose a new setup folder; existing files were retained')
            return
        temp = destination.with_name('source.partial-' + uuid.uuid4().hex)
        temp.mkdir()
        try:
            for item, name in members:
                file = temp / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(z.read(item))
                if os.name != 'nt':
                    file.chmod(0o755 if files[name][0] == '100755' else 0o644)
            temp.rename(destination)
        except BaseException:
            shutil.rmtree(temp)  # only the temporary directory created by this call
            raise


class Installer:
    def __init__(self, root, compute, approved=False, parent_pid=None):
        if not supported():
            raise ValueError('Automatic setup currently supports Windows x64 and Linux x86-64')
        if not approved:
            raise ValueError('Approve local setup and its downloads in Blender before installing')
        self.p = paths(root, compute)
        self.compute = compute
        self.parent_pid = parent_pid
        self.env = clean_environment(self.p, compute)
        self.child = None

    def cancelled(self):
        if self.p['cancel'].exists():
            raise Cancelled('Setup cancelled. Cached downloads retained; retry explicitly to continue')
        if self.parent_pid and not parent_alive(self.parent_pid):
            raise Cancelled('Blender closed; setup stopped')

    def status(self, state, message, **extra):
        atomic_json(self.p['status'], dict(state=state, message=message, **extra))
        print(message, flush=True)

    def download(self, url, file, sha256=None, limit=512*1024**2):
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        if file.is_file() and (sha256 is None or hashlib.sha256(file.read_bytes()).hexdigest() == sha256):
            return file
        self.cancelled()
        temp = file.with_suffix(file.suffix + '.part')
        digest, size = hashlib.sha256(), 0
        req = Request(url, headers={'User-Agent': 'KimodoMotionStudio/0.4.0'})
        with urlopen(req, timeout=30) as response, temp.open('wb') as output:
            if not response.geturl().startswith('https://'):
                raise ValueError('HTTPS is required for setup downloads')
            while True:
                self.cancelled()
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ValueError('Download exceeds safety limit')
                output.write(chunk); digest.update(chunk)
        if sha256 and digest.hexdigest() != sha256:
            temp.unlink(missing_ok=True)
            raise ValueError('Setup download failed SHA-256 verification')
        os.replace(temp, file)
        return file

    def command(self, args, *, cwd=None, env=None):
        self.cancelled()
        print('RUN:', ' '.join(map(str, args)), flush=True)
        kwargs = dict(cwd=str(cwd or self.p['home']), env=env or self.env,
                      stdin=subprocess.DEVNULL, shell=False)
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['start_new_session'] = True
        self.child = subprocess.Popen(list(map(str, args)), **kwargs)
        try:
            while self.child.poll() is None:
                self.cancelled(); time.sleep(.2)
            if self.child.returncode:
                raise RuntimeError(f'Setup command failed ({self.child.returncode}). See setup.log for the exact dependency error')
        except BaseException:
            stop_child(self.child)
            raise
        finally:
            self.child = None

    def bootstrap_uv(self):
        url, checksum = UV_ASSETS[platform.system()]
        wheel = self.download(url, self.p['home'] / 'cache/uv-bootstrap.whl', checksum)
        # Re-extract from verified archive even on retry; do not trust a stale executable.
        with zipfile.ZipFile(wheel) as z:
            binary_name = 'uv.exe' if os.name == 'nt' else 'uv'
            hits = [i for i in z.infolist() if not i.is_dir() and PurePosixPath(i.filename).name == binary_name]
            if len(hits) != 1 or hits[0].file_size > 150*1024**2:
                raise ValueError('Unexpected uv bootstrap wheel structure')
            target = self.p['home'] / 'tools' / binary_name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(z.read(hits[0]))
            if os.name != 'nt':
                target.chmod(0o700)
        return target

    def install(self):
        own_runtime(self.p)
        with exclusive_lock(self.p['lock']):
            self.p['ready'].unlink(missing_ok=True)
            self.status('installing', '1/6 — Downloading verified private installer')
            uv = self.bootstrap_uv()
            self.status('installing', '2/6 — Preparing isolated Python 3.11 (not Blender Python)')
            self.command([uv, '--no-config', 'python', 'install', PYTHON_VERSION])
            if not self.p['python'].is_file():
                self.command([uv, '--no-config', 'venv', '--managed-python', '--python', PYTHON_VERSION,
                              '--seed', self.p['venv']])
            pip = [uv, '--no-config', 'pip', 'install', '--python', self.p['python']]
            self.status('installing', '3/6 — Downloading and verifying pinned Kimodo source')
            archive = self.download(SOURCE_URL, self.p['home'] / 'cache/kimodo-source.zip')
            try:
                install_source(archive, self.p['source'])
            except (zipfile.BadZipFile, ValueError):
                archive.unlink(missing_ok=True)  # Retry obtains a fresh, verified archive.
                raise
            self.status('installing', '4/6 — Installing local PyTorch ' + self.compute + ' runtime')
            self.command(pip + ['--index-url', 'https://download.pytorch.org/whl/' + self.compute,
                               '--only-binary', ':all:', 'torch==' + TORCH_VERSION])
            self.status('installing', '5/6 — Installing Kimodo dependencies (model weights are separate)')
            import tomllib
            metadata = tomllib.loads((self.p['source'] / 'pyproject.toml').read_text())
            requirements = self.p['home'] / 'requirements.txt'
            requirements.write_text('\n'.join(metadata['project']['dependencies']) +
                                    '\nsetuptools==75.8.2\nwheel==0.45.1\n', encoding='utf-8')
            constraints = self.p['home'] / 'constraints.txt'
            constraints.write_text('torch==' + TORCH_VERSION + '\ntransformers==5.1.0\n', encoding='utf-8')
            self.command(pip + ['--index-url', 'https://pypi.org/simple', '--only-binary', ':all:',
                               '--constraint', constraints, '--requirement', requirements])
            # C++ MotionCorrection is optional. Never install compilers or elevation tools.
            self.command(pip + ['--no-deps', '--no-build-isolation', self.p['source']])
            self.status('verifying', '6/6 — Checking backend imports, skeleton assets and compute device')
            report = self.p['home'] / 'probe.json'
            report.unlink(missing_ok=True)
            self.command([self.p['python'], '-I', Path(__file__).resolve(), '--probe', report,
                          '--compute', self.compute], env=dict(self.env, HF_HUB_OFFLINE='1',
                              TRANSFORMERS_OFFLINE='1', TEXT_ENCODER_MODE='local'))
            data = read_json(report)
            if data.get('ok') is not True:
                raise RuntimeError('Backend verification did not produce a success report')
            self.cancelled()
            atomic_json(self.p['ready'], dict(profile=PROFILE, source_commit=SOURCE_COMMIT,
                        home=str(self.p['home']), compute=self.compute, verified=True, probe=data))
            self.status('ready', 'Local backend ready. Choose or download model weights in Generate', probe=data)


def probe(path, compute):
    """No model downloads or inference. Test real imports, FK assets and a tensor op."""
    import importlib.util
    import torch
    import kimodo
    import transformers
    import peft
    from kimodo.skeleton.registry import build_skeleton
    from kimodo.exports.motion_io import load_motion_file
    from kimodo.constraints import load_constraints_lst
    from kimodo.model.llm2vec import LLM2VecEncoder
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Managed backend must use Python 3.11')
    if torch.__version__.split('+')[0] != TORCH_VERSION:
        raise RuntimeError('Unexpected torch version')
    for count in (77, 30, 34, 22):
        sk = build_skeleton(count)
        sk.fk(torch.eye(3).repeat(1, count, 1, 1), torch.zeros(1, 3))
    if compute == 'cu128' and not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA is unavailable. Check your NVIDIA driver or explicitly choose the CPU setup profile')
    device = 'cuda:0' if compute == 'cu128' else 'cpu'
    result = (torch.ones((8, 8), device=device) @ torch.ones((8, 8), device=device)).sum().item()
    if result != 512.:
        raise RuntimeError('Compute verification failed')
    from importlib.metadata import distributions
    packages = {d.metadata['Name']: d.version for d in distributions() if d.metadata.get('Name')}
    atomic_json(path, dict(ok=True, python=sys.version, torch=torch.__version__, packages=packages,
                          transformers=transformers.__version__, peft=peft.__version__, device=device,
                          gpu=torch.cuda.get_device_name(0) if compute == 'cu128' else None,
                          motion_correction=importlib.util.find_spec('motion_correction') is not None,
                          real_model_inference='not tested', model_weights='not downloaded'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root'); parser.add_argument('--compute', choices=['cu128', 'cpu'], default='cu128')
    parser.add_argument('--approved', action='store_true')
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--probe')
    argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if args.probe:
        probe(args.probe, args.compute); return 0
    if not args.root:
        parser.error('--root is required')
    installer = None
    try:
        installer = Installer(args.root, args.compute, args.approved, args.parent_pid)
        installer.install()
    except BaseException as exc:
        traceback.print_exc()
        if installer and installer.p['home'].is_dir() and not isinstance(exc, SetupBusy):
            # Do not report errors into an unowned directory.
            if read_json(installer.p['owner']).get('application') == 'KimodoMotionStudio':
                installer.status('cancelled' if isinstance(exc, Cancelled) else 'error', str(exc))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
