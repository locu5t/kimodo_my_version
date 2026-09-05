"""Networked CI integration test. Installs CPU backend, NEVER model weights.

Run only deliberately: python blender_tools/managed_install_smoke.py
Uses a disposable directory selected by KIMODO_SETUP_TEST_ROOT / OS temp.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'blender_addon'))
from kimodo_motion_studio import managed_setup as ms
location=Path(os.environ.get('KIMODO_SETUP_TEST_ROOT',str(Path(tempfile.gettempdir())/'kimodo-managed-ci')))
location.mkdir(parents=True,exist_ok=True)
report={'root':str(location),'profile':ms.PROFILE,'model_weights':'not downloaded'}
try:
    command=[sys.executable,str(Path(ms.__file__).resolve()),'--root',str(location),'--compute','cpu','--approved']
    with (location/'integration.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
    print((location/'integration.log').read_text(encoding='utf-8',errors='replace'))
    assert result.returncode==0,'Managed setup failed; see integration.log'
    assert ms.ready(location,'cpu'),'No valid completion marker'
    p=ms.paths(location,'cpu')
    report.update(status='passed',probe=ms.read_json(p['home']/'probe.json'))
    # Second pass uses existing runtime/download cache and re-verifies it.
    result=subprocess.run(command,check=False)
    assert result.returncode==0 and ms.ready(location,'cpu'),'Retry/reuse failed'
    report['reuse']='passed'
except BaseException as exc:
    report.update(status='failed',error=str(exc))
    raise
finally:
    (root/'managed-install-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
