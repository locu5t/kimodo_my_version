"""Local-storage and mocked-download tests; do not download real model weights."""
import json
import os
from pathlib import Path
import sys
import types
import pytest
from kimodo_motion_studio import model_storage as ms

CONFIG = '''_target_: kimodo.model.Kimodo
denoiser:
  ckpt_path: ${oc.select:checkpoint_dir}/model.safetensors
  motion_rep:
    stats_path: ${oc.select:checkpoint_dir}/stats/motion/
'''


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setattr(os, 'environ', dict(os.environ))


def checkpoint(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'config.yaml').write_text(CONFIG)
    (folder/'model.safetensors').write_bytes(b'fixture-not-real-weights')
    for group in ('body', 'global_root', 'local_root'):
        p = folder/'stats'/'motion'/group
        p.mkdir(parents=True, exist_ok=True)
        for name in ('mean.npy', 'std.npy'): (p/name).write_bytes(b'fixture')
    return folder


def text_fixture(folder, repo):
    folder.mkdir(parents=True, exist_ok=True)
    names = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    names += ['model.safetensors'] if repo == ms.TEXT_REPOS[0] else ['adapter_config.json', 'adapter_model.safetensors']
    for name in names: (folder/name).write_text('{}')
    return folder


def request(tmp_path, operation='check_models', mode='MANAGED'):
    return {'schema_version': 1, 'operation': operation, 'model': ms.MODEL_NAMES[0],
            'storage': {'mode': mode, 'models_root': str(tmp_path/'models'),
                        'manual_path': str(tmp_path/'custom'/'arbitrary name')}}


@pytest.mark.parametrize('model', ms.MODEL_NAMES)
def test_presets_have_unique_readable_local_folders(tmp_path, model):
    req = request(tmp_path)
    folder = ms.checkpoint_folder(req['storage'], model)
    assert folder == tmp_path/'models'/model
    assert ms.repository_id(model) == 'nvidia/'+model


@pytest.mark.parametrize('bad', ['../evil', 'owner/../bad', 'https://host/model', r'C:\model', 'owner/--bad', '', 'foo'])
def test_invalid_download_identifiers(bad):
    with pytest.raises(ValueError): ms.repository_id(bad)


def test_custom_namespace_cannot_collide(tmp_path):
    assert ms.checkpoint_folder(request(tmp_path)['storage'], 'alice/Kimodo-SOMA-RP-v1.1').name == 'alice__Kimodo-SOMA-RP-v1.1'


def test_no_silent_default_download_folder():
    with pytest.raises(ValueError, match='Choose'): ms.checkpoint_folder({}, ms.MODEL_NAMES[0])


def test_manual_arbitrary_name_used_exactly_and_not_downloaded(tmp_path):
    req = request(tmp_path, mode='MANUAL')
    assert ms.checkpoint_folder(req['storage'], req['model']) == tmp_path/'custom'/'arbitrary name'
    with pytest.raises(ValueError, match='read-only'):
        ms.checkpoint_folder(req['storage'], req['model'], download=True)


def test_relative_manual_path_rejected():
    with pytest.raises(ValueError, match='absolute'):
        ms.checkpoint_folder({'mode': 'MANUAL', 'manual_path': 'relative'}, 'anything')


def test_destination_symlink_escape_rejected(tmp_path):
    req = request(tmp_path)
    root = Path(req['storage']['models_root']); root.mkdir()
    target = tmp_path/'elsewhere'; target.mkdir()
    try: (root/req['model']).symlink_to(target, target_is_directory=True)
    except OSError: pytest.skip('Symlinks not permitted by this OS')
    with pytest.raises(ValueError, match='escape'): ms.checkpoint_folder(req['storage'], req['model'])


def test_cache_is_beneath_chosen_root_unless_explicit(tmp_path):
    s = request(tmp_path)['storage']
    assert ms.cache_home(s) == tmp_path/'models'/'.huggingface'
    s['cache_root'] = str(tmp_path/'existing-hf')
    assert ms.cache_home(s) == tmp_path/'existing-hf'


def test_generation_offline_even_with_old_download_flag(tmp_path):
    req = request(tmp_path, 'generate'); req['allow_downloads'] = True
    os.environ.update(HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0', TEXT_ENCODER_MODE='api')
    ms.configure_environment(req)
    assert os.environ['HF_HUB_OFFLINE'] == '1'
    assert os.environ['TRANSFORMERS_OFFLINE'] == '1'
    assert os.environ['TEXT_ENCODER_MODE'] == 'local'
    assert os.environ['HF_HUB_CACHE'] == str(tmp_path/'models'/'.huggingface'/'hub')


def test_download_requires_explicit_authorization(tmp_path):
    req = request(tmp_path, 'download_model')
    with pytest.raises(ValueError, match='confirmation'): ms.configure_environment(req)
    req['download_confirmed'] = True
    ms.configure_environment(req)
    assert os.environ['HF_HUB_OFFLINE'] == '0'
    req['operation'] = 'generate'
    ms.configure_environment(req)
    assert os.environ['HF_HUB_OFFLINE'] == '1'


def test_relocating_cache_preserves_hf_login_location(tmp_path):
    os.environ['HF_HOME'] = str(tmp_path/'original')
    os.environ.pop('HF_TOKEN_PATH', None)
    ms.configure_environment(request(tmp_path))
    assert os.environ['HF_TOKEN_PATH'] == str(tmp_path/'original'/'token')


def test_existing_token_override_preserved(tmp_path):
    os.environ['HF_TOKEN_PATH'] = '/custom/token'
    ms.configure_environment(request(tmp_path))
    assert os.environ['HF_TOKEN_PATH'] == '/custom/token'


def test_local_file_check_finds_complete_checkpoint(tmp_path):
    folder = checkpoint(tmp_path/'checkpoint')
    _, files = ms.checkpoint_config(folder)
    assert len(files) == 7


@pytest.mark.parametrize('missing', ['config.yaml', 'model.safetensors', 'stats/motion/local_root/std.npy', 'stats/motion/global_root/mean.npy'])
def test_partial_checkpoint_rejected(tmp_path, missing):
    folder = checkpoint(tmp_path/'checkpoint'); (folder/missing).unlink()
    with pytest.raises(ValueError): ms.checkpoint_config(folder)


def test_lfs_pointer_is_not_a_model(tmp_path):
    folder = checkpoint(tmp_path/'checkpoint')
    (folder/'model.safetensors').write_text('version https://git-lfs.github.com/spec/v1\noid sha256:bad')
    with pytest.raises(ValueError, match='LFS'): ms.checkpoint_config(folder)


def test_interrupted_download_blocked_until_verified(tmp_path):
    folder = checkpoint(tmp_path/'checkpoint')
    (folder/ms.STATE_FILE).write_text('{"state":"downloading"}')
    with pytest.raises(ValueError, match='interrupted'): ms.checkpoint_config(folder)
    ms.checkpoint_config(folder, ignore_pending=True)


def test_unsafe_config_target_rejected(tmp_path):
    folder = checkpoint(tmp_path/'checkpoint')
    (folder/'config.yaml').write_text('_target_: os.system\n')
    with pytest.raises(ValueError, match='not a Kimodo'): ms.checkpoint_config(folder)


def test_text_shard_index_detects_missing_shard(tmp_path):
    folder = text_fixture(tmp_path/'llama', ms.TEXT_REPOS[0])
    (folder/'model.safetensors.index.json').write_text('{"weight_map":{"a":"part-1.safetensors","b":"part-2.safetensors"}}')
    (folder/'part-1.safetensors').write_bytes(b'fixture')
    with pytest.raises(ValueError, match='part-2'): ms.validate_text_snapshot(folder, ms.TEXT_REPOS[0])


def test_local_check_never_uses_network(tmp_path, monkeypatch):
    req = request(tmp_path); checkpoint(ms.checkpoint_folder(req['storage'], req['model']))
    calls = []
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(text_fixture(tmp_path/kwargs['repo_id'], kwargs['repo_id']))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=snapshot_download))
    result = ms.inspect_storage(req)
    assert result['ready'] and len(result['components']) == 4
    assert len(calls) == 3 and all(c['local_files_only'] is True for c in calls)


def test_download_only_selected_motion_unless_text_requested(tmp_path, monkeypatch):
    req = request(tmp_path, 'download_model'); req['download_confirmed'] = True
    calls = []
    def download(**kw):
        calls.append(kw)
        return checkpoint(Path(kw['local_dir']))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=download))
    ms.download_assets(req, lambda *args: None)
    assert len(calls) == 1
    assert calls[0]['repo_id'] == 'nvidia/'+req['model']
    state = ms.checkpoint_folder(req['storage'], req['model'])/ms.STATE_FILE
    assert json.loads(state.read_text())['state'] == 'complete'
    assert 'force_download' not in calls[0]


def test_download_includes_actual_llama_base_and_both_adapters(tmp_path, monkeypatch):
    req = request(tmp_path, 'download_model')
    req.update(download_confirmed=True, include_text_encoder=True)
    calls = []
    def download(**kw):
        calls.append(kw)
        if 'local_dir' in kw: return checkpoint(Path(kw['local_dir']))
        return text_fixture(Path(kw['cache_dir'])/kw['repo_id'], kw['repo_id'])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=download))
    ms.download_assets(req, lambda *args: None)
    assert [c['repo_id'] for c in calls[1:]] == list(ms.TEXT_REPOS)
    assert all(c['cache_dir'].startswith(str(tmp_path)) for c in calls[1:])


def test_failed_download_is_not_marked_ready(tmp_path, monkeypatch):
    req = request(tmp_path, 'download_model'); req['download_confirmed'] = True
    def fail(**kw): raise OSError('disk full')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=fail))
    with pytest.raises(OSError): ms.download_assets(req, lambda *args: None)
    state = ms.checkpoint_folder(req['storage'], req['model'])/ms.STATE_FILE
    assert json.loads(state.read_text())['state'] == 'downloading'


def test_arbitrary_manual_folder_passed_into_local_model_config(tmp_path, monkeypatch):
    req = request(tmp_path, 'generate', mode='MANUAL')
    folder = checkpoint(Path(req['storage']['manual_path']))
    seen = {}
    class Model:
        def eval(self): seen['eval'] = True; return self
    def instantiate(cfg, overrides):
        seen.update(cfg=cfg, overrides=overrides); return Model()
    for name, attrs in {'kimodo': {}, 'kimodo.model': {}, 'kimodo.model.loading': {'instantiate_from_dict': instantiate}}.items():
        module = types.ModuleType(name); module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    ms.load_selected_model(req)
    assert seen['cfg']['denoiser']['ckpt_path'] == str(folder/'model.safetensors')
    assert seen['cfg']['text_encoder']['_target_'] == 'kimodo.model.LLM2VecEncoder'
    assert seen['eval'] and 'checkpoint_dir' not in seen['cfg']


def test_storage_jobs_do_not_load_torch_models(tmp_path, monkeypatch):
    from kimodo_motion_studio import engine
    req = request(tmp_path)
    (tmp_path/'request.json').write_text(json.dumps(req))
    monkeypatch.setattr(engine, 'inspect_storage', lambda r: {'ready': False, 'components': []})
    def forbidden(): raise AssertionError('adapter should not load for storage operations')
    monkeypatch.setattr(engine, 'KimodoAdapter', forbidden)
    assert engine.run_job(tmp_path)['state'] == 'done'
