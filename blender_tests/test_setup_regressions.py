"""Regressions discovered by real Windows/Linux installation CI."""
from kimodo_motion_studio import managed_setup as ms


def test_windows_child_logging_is_utf8(tmp_path):
    env = ms.clean_environment(ms.paths(tmp_path, 'cpu'), 'cpu')
    assert env['PYTHONIOENCODING'] == 'utf-8'


def test_pure_python_antlr_installed_before_wheel_only_resolution(tmp_path, monkeypatch):
    installer = ms.Installer(tmp_path, 'cpu', True)
    calls = []
    monkeypatch.setattr(installer, 'bootstrap_uv', lambda: tmp_path / 'uv')
    monkeypatch.setattr(installer, 'download', lambda *a, **k: tmp_path / 'archive')
    def source(*args, **kwargs):
        (installer.p['source'] / 'kimodo').mkdir(parents=True)
        (installer.p['source'] / 'kimodo/__init__.py').write_text('# fixture')
        (installer.p['source'] / 'pyproject.toml').write_text('[project]\ndependencies=[]')
    monkeypatch.setattr(ms, 'install_source', source)
    def command(args, **kwargs):
        calls.append([str(x) for x in args])
        if 'venv' in args:
            installer.p['python'].parent.mkdir(parents=True)
            installer.p['python'].write_text('fixture')
        if '--probe' in args:
            ms.atomic_json(args[args.index('--probe') + 1], {'ok': True})
    monkeypatch.setattr(installer, 'command', command)
    installer.install()
    antlr = next(c for c in calls if 'antlr4-python3-runtime==4.9.3' in c)
    deps = next(c for c in calls if '--requirement' in c)
    assert '--only-binary' not in antlr
    assert '--only-binary' in deps and calls.index(antlr) < calls.index(deps)
