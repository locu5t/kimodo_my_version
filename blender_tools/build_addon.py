#!/usr/bin/env python3
"""Build a Blender extension ZIP (manifest at archive root)."""
from pathlib import Path
import argparse
import zipfile

def build(output=None):
    root=Path(__file__).resolve().parents[1]
    addon=root/"blender_addon"/"kimodo_motion_studio"
    output=Path(output) if output else root/"blender_dist"/"kimodo_motion_studio-0.4.0.zip"
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(addon.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                z.write(file,file.relative_to(addon).as_posix())
    return output

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output")
    print(build(parser.parse_args().output))
