# SPDX-License-Identifier: GPL-3.0-or-later
"""Execute with the Kimodo venv Python, NOT Blender's bundled Python."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--kimodo-root", required=True)
    args = parser.parse_args()
    folder, root = Path(args.job).resolve(), Path(args.kimodo_root).resolve()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "worker_pid.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        if not (root / "kimodo" / "__init__.py").is_file():
            raise ValueError("Kimodo repository path must contain kimodo/__init__.py")
        here = Path(__file__).resolve().parent
        # Blender extensions may install under a qualified module name. This private
        # package loads pure bridge modules without importing bpy or relying on that name.
        spec = importlib.util.spec_from_file_location("_kimodo_bridge_worker", here / "__init__.py",
                                                      submodule_search_locations=[str(here)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)
        sys.path.insert(0, str(root))
        from _kimodo_bridge_worker.engine import run_job
        os.chdir(root)
        run_job(folder)
    except Exception as exc:
        traceback.print_exc()
        folder.mkdir(parents=True, exist_ok=True)
        status = {"state": "error", "message": str(exc), "traceback": traceback.format_exc()}
        temp = folder / "status.json.tmp"
        temp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        os.replace(temp, folder / "status.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
