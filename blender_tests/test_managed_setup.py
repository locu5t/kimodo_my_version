"""Installer invariants; fixture downloads/commands are simulated, not real installs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
from kimodo_motion_studio import managed_setup as ms


def make_archive(path, items=None):
    items = items or {'kimodo/__init__.py': b'# Kimodo fixture\n', 'pyproject.toml': b'[project]\ndependencies=[]\n'}
    files = {}
    with zipfile.ZipFile(path, 'w') as z:
        for name, content in items.items():
            info = zipfile.ZipInfo('repo-fixture/' + name)
            info.external_attr = 0o100644 << 16
            z.writestr(info, content)
            files[name] = ('100644', ms.git_object('blob', content))
    return ms.git_tree(files)


@pytest.mark.parametrize('system,machine,yes', [('Windows','AMD64',True),('Linux','x86_64',True),('Linux','aarch64',False),('Darwin','arm64',False)])
def test_supported(system, machine, yes):
    assert ms.supported(system,machine) is yes


@pytest.mark.parametrize('system,exe', [('Windows','Scripts/python.exe'),('Linux','bin/python')])
def test_runtime_layout(tmp_path,system,exe):
    p=ms.paths(tmp_path,'cpu',system)
    assert p['python']==p['venv']/exe and p['home'].is_relative_to(tmp_path)
    assert ms.PROFILE in str(p['home'])
    assert ms.paths(tmp_path,'cu128',system)['home']!=p['home']


def test_profile_no_arbitrary_command(tmp_path):
    with pytest.raises(ValueError): ms.paths(tmp_path,'cpu; rm -rf ~')
    with pytest.raises(ValueError,match='Approve'): ms.Installer(tmp_path,'cpu')


def test_unowned_directory_not_touched(tmp_path):
    p=ms.paths(tmp_path,'cpu');p['home'].mkdir(parents=True)
    keep=p['home']/'existing.txt';keep.write_text('keep')
    with pytest.raises(ValueError,match='unowned'): ms.own_runtime(p)
    assert keep.read_text()=='keep' and not p['owner'].exists()


def test_own_runtime_idempotent(tmp_path):
    p=ms.paths(tmp_path,'cpu');ms.own_runtime(p);ms.own_runtime(p)
    assert ms.read_json(p['owner'])['profile']==ms.PROFILE


def test_runtime_symlink_rejected(tmp_path):
    p=ms.paths(tmp_path,'cpu');other=tmp_path/'other';other.mkdir()
    p['home'].parent.mkdir(parents=True)
    try: p['home'].symlink_to(other,target_is_directory=True)
    except OSError: pytest.skip('symlinks unavailable')
    with pytest.raises(ValueError,match='symlink'):ms.own_runtime(p)
    assert not list(other.iterdir())


def test_environment_does_not_pollute_blender(tmp_path,monkeypatch):
    monkeypatch.setenv('PYTHONPATH','wrong');monkeypatch.setenv('PYTHONHOME','wrong')
    monkeypatch.setenv('VIRTUAL_ENV','wrong');monkeypatch.setenv('UV_PYTHON','wrong')
    monkeypatch.setenv('PIP_TARGET','wrong');monkeypatch.setenv('HF_TOKEN','fake-test-token')
    before=dict(os.environ);p=ms.paths(tmp_path,'cpu');env=ms.clean_environment(p,'cpu')
    assert os.environ==before
    assert all(k not in env for k in ('PYTHONPATH','PYTHONHOME','PIP_TARGET','VIRTUAL_ENV','UV_PYTHON'))
    assert env['UV_PYTHON_INSTALL_REGISTRY']=='false' and env['UV_PYTHON_INSTALL_BIN']=='false'
    assert env['SKIP_MOTION_CORRECTION_IN_SETUP']=='1'
    assert env['UV_CACHE_DIR'].startswith(str(p['home']))


def test_os_lock_released(tmp_path):
    p=tmp_path/'lock'
    with ms.exclusive_lock(p):
        with pytest.raises(RuntimeError,match='Another setup'):
            with ms.exclusive_lock(p):pass
    with ms.exclusive_lock(p):pass


def test_archive_git_hash_matches_git(tmp_path):
    """Use the real Git executable as an independent hashing oracle."""
    import shutil
    if not shutil.which('git'):pytest.skip('git unavailable')
    source=tmp_path/'git';source.mkdir()
    subprocess.run(['git','init',str(source)],check=True,capture_output=True)
    files={'foo.txt':b'a\n','foo/x':b'b','foo.y':b'c'}
    for n,c in files.items():
        p=source/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(c)
    subprocess.run(['git','-C',str(source),'add','.'],check=True,capture_output=True)
    expected=subprocess.check_output(['git','-C',str(source),'write-tree'],text=True).strip()
    assert ms.git_tree({n:('100644',ms.git_object('blob',c)) for n,c in files.items()})==expected


def test_source_verified_before_extract(tmp_path):
    arc=tmp_path/'s.zip';digest=make_archive(arc);dest=tmp_path/'source'
    with pytest.raises(ValueError,match='pinned Git tree'):ms.install_source(arc,dest,'0'*40)
    assert not dest.exists()
    ms.install_source(arc,dest,digest)
    assert (dest/'kimodo/__init__.py').exists()
    ms.install_source(arc,dest,digest)  # retry reuses an intact tree
    (dest/'kimodo/__init__.py').write_text('user edit')
    with pytest.raises(ValueError,match='modified'):ms.install_source(arc,dest,digest)
    assert (dest/'kimodo/__init__.py').read_text()=='user edit'


@pytest.mark.parametrize('bad', ['../escape','/escape','repo/../escape','repo/a\\b','repo/C:escape','repo/.git/config'])
def test_unsafe_source_paths(tmp_path,bad):
    arc=tmp_path/'bad.zip'
    with zipfile.ZipFile(arc,'w') as z:z.writestr(bad,b'bad')
    with zipfile.ZipFile(arc) as z:
        with pytest.raises(ValueError):ms.source_members(z)


def test_archive_symlinks_rejected(tmp_path):
    arc=tmp_path/'bad.zip'
    with zipfile.ZipFile(arc,'w') as z:
        i=zipfile.ZipInfo('root/link');i.external_attr=0o120777<<16;z.writestr(i,b'/etc/passwd')
    with zipfile.ZipFile(arc) as z:
        with pytest.raises(ValueError,match='Links'):ms.source_members(z)


def test_missing_files_cannot_be_ready(tmp_path):
    p=ms.paths(tmp_path,'cpu');ms.own_runtime(p)
    ms.atomic_json(p['ready'],dict(profile=ms.PROFILE,source_commit=ms.SOURCE_COMMIT,compute='cpu',verified=True,home=str(p['home'])))
    assert not ms.ready(tmp_path,'cpu')
    p['python'].parent.mkdir(parents=True);p['python'].write_text('fixture')
    (p['source']/'kimodo').mkdir(parents=True);(p['source']/'kimodo/__init__.py').write_text('#fixture')
    assert ms.ready(tmp_path,'cpu')
    assert not ms.ready(tmp_path,'cu128')


def test_cancel_before_network(tmp_path,monkeypatch):
    i=ms.Installer(tmp_path,'cpu',True);ms.own_runtime(i.p)
    i.p['cancel'].write_text('cancel')
    monkeypatch.setattr(ms,'urlopen',lambda *a,**k:pytest.fail('network accessed'))
    with pytest.raises(ms.Cancelled):i.download('https://example.invalid',tmp_path/'new')


def test_cached_uv_requires_checksum(tmp_path,monkeypatch):
    f=tmp_path/'uv.whl';f.write_bytes(b'verified')
    i=ms.Installer(tmp_path,'cpu',True)
    monkeypatch.setattr(ms,'urlopen',lambda *a,**k:pytest.fail('network accessed'))
    assert i.download('https://example.invalid',f,hashlib.sha256(b'verified').hexdigest())==f


def test_setup_subprocess_cancel_terminates_child(tmp_path,monkeypatch):
    i=ms.Installer(tmp_path,'cpu',True);ms.own_runtime(i.p)
    calls=[]
    class Proc:
        def poll(self):return None
    monkeypatch.setattr(ms.subprocess,'Popen',lambda *a,**k:Proc())
    monkeypatch.setattr(ms,'stop_child',lambda p:calls.append(p))
    count=[0]
    def cancel():
        count[0]+=1
        if count[0]>1:raise ms.Cancelled('test')
    monkeypatch.setattr(i,'cancelled',cancel)
    with pytest.raises(ms.Cancelled):i.command(['fixture'])
    assert len(calls)==1


def test_simulated_install_only_ready_after_probe(tmp_path,monkeypatch):
    i=ms.Installer(tmp_path,'cpu',True)
    calls=[]
    monkeypatch.setattr(i,'bootstrap_uv',lambda:tmp_path/'uv')
    monkeypatch.setattr(i,'download',lambda *a,**k:tmp_path/'archive')
    def install_source(*a,**k):
        (i.p['source']/'kimodo').mkdir(parents=True)
        (i.p['source']/'kimodo/__init__.py').write_text('#fixture')
        (i.p['source']/'pyproject.toml').write_text('[project]\ndependencies=["numpy>=1.23"]')
    monkeypatch.setattr(ms,'install_source',install_source)
    def command(args,**kwargs):
        calls.append(list(map(str,args)))
        assert not i.p['ready'].exists()
        if 'venv' in args:
            i.p['python'].parent.mkdir(parents=True);i.p['python'].write_text('fixture')
        if '--probe' in args:
            ms.atomic_json(args[args.index('--probe')+1],{'ok':True})
    monkeypatch.setattr(i,'command',command)
    i.install()
    assert ms.ready(tmp_path,'cpu')
    assert all('--python' in c and str(i.p['python']) in c for c in calls if 'pip' in c)
    assert all('--system' not in c for c in calls)
    assert any('https://download.pytorch.org/whl/cpu' in c for c in calls)
    assert not any('git' in c[0] for c in calls)


def test_failed_probe_cannot_mark_ready(tmp_path,monkeypatch):
    i=ms.Installer(tmp_path,'cpu',True)
    monkeypatch.setattr(i,'bootstrap_uv',lambda:(_ for _ in ()).throw(RuntimeError('offline')))
    with pytest.raises(RuntimeError):i.install()
    assert not i.p['ready'].exists()


def test_sha_pins_are_complete():
    assert len(ms.SOURCE_TREE)==40 and len(ms.SOURCE_COMMIT)==40
    for url,sha in ms.UV_ASSETS.values():
        assert url.startswith('https://files.pythonhosted.org/') and '0.8.22' in url
        assert len(sha)==64 and int(sha,16)>0
