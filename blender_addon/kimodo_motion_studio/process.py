# SPDX-License-Identifier: GPL-3.0-or-later
"""Subprocess launch helpers; no shell interpolation and no Blender dependency."""
import os
from pathlib import Path
import re
import subprocess


def wsl_path(path):
    path = str(path)
    if path.startswith("/"):
        return path
    match = re.match(r"^([a-zA-Z]):[\\/](.*)$", path)
    if not match:
        raise ValueError("WSL bridge files must be on a local Windows drive, not a UNC/network path")
    return "/mnt/" + match.group(1).lower() + "/" + match.group(2).replace("\\", "/")


def build_command(python, repository, worker, job_dir, mode="NATIVE", distro="Ubuntu-22.04"):
    if not python.strip() or not repository.strip():
        raise ValueError("Set the external Python executable and Kimodo repository in add-on preferences")
    if mode == "WSL":
        if not python.startswith("/"):
            raise ValueError("WSL Python must be an absolute Linux path to the Kimodo venv Python")
        return ["wsl.exe", "--distribution", distro, "--exec", python, wsl_path(worker),
                "--job", wsl_path(job_dir), "--kimodo-root", wsl_path(repository)]
    if mode != "NATIVE":
        raise ValueError("Unknown backend mode")
    if not Path(python).is_file():
        raise ValueError("External Python executable does not exist")
    return [python, str(worker), "--job", str(job_dir), "--kimodo-root", str(repository)]


def launch(command, log_file):
    kwargs = {"stdout": log_file, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
              "shell": False, "env": dict(os.environ, PYTHONUNBUFFERED="1")}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(command, **kwargs)
