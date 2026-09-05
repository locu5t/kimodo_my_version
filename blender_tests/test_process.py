from pathlib import Path
import subprocess
import sys
import json
import pytest
from kimodo_motion_studio.process import wsl_path, build_command

@pytest.mark.parametrize("win,expected", [
    (r"E:\Creative Tech\kimodo", "/mnt/e/Creative Tech/kimodo"),
    ("C:/Users/user name/bridge.py", "/mnt/c/Users/user name/bridge.py"),
    ("/home/user/venv/bin/python", "/home/user/venv/bin/python")])
def test_wsl(win,expected):
    assert wsl_path(win)==expected


def test_unc_rejected():
    with pytest.raises(ValueError):
        wsl_path(r"\\server\share\file")


def test_command_paths_are_distinct_arguments(tmp_path):
    command=build_command(sys.executable,str(tmp_path),"a; weird worker.py","job & folder")
    assert command[1]=="a; weird worker.py"
    assert command[3]=="job & folder"
    assert len(command)==6


def test_wsl_command():
    command=build_command("/home/user/venv/bin/python",r"E:\kimodo",r"C:\User\worker.py",r"E:\jobs\1",
                          "WSL","Ubuntu-22.04")
    assert command==["wsl.exe","--distribution","Ubuntu-22.04","--exec","/home/user/venv/bin/python",
                     "/mnt/c/User/worker.py","--job","/mnt/e/jobs/1","--kimodo-root","/mnt/e/kimodo"]


def test_worker_reports_missing_repository(tmp_path):
    worker=Path(__file__).resolve().parents[1]/"blender_addon/kimodo_motion_studio/worker.py"
    result=subprocess.run([sys.executable,str(worker),"--job",str(tmp_path),"--kimodo-root",str(tmp_path/"missing")],
                          capture_output=True,text=True)
    assert result.returncode==1
    state=json.loads((tmp_path/"status.json").read_text())
    assert state["state"]=="error"
    assert "repository path" in state["message"]
