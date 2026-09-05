# Kimodo Motion Studio 0.4.0 — managed local setup

Install the **0.4.0 extension ZIP**, not a source ZIP. Enable it in Blender 5.2.
On a fresh installation the add-on automatically opens **Set up local Kimodo**.
Choose the installation folder and NVIDIA/CPU profile, then click **OK** once to
approve the downloads. The rest of backend setup is automatic. No terminal,
separate Python installation, Git, manually created venv or bridge server is
required. Installing the small ZIP is not itself a completed backend/model download.

Blender's **Allow Online Access** preference must be enabled. Setup does not
change that setting. Cancelling the dialog leaves the authoring UI available and
starts no downloads. Reopen it from **N > Kimodo > Kimodo Local Backend** or the
add-on preferences. Existing configured 0.3.x environments are kept in advanced
external mode; select **Automatic local setup** to migrate deliberately.

## What is installed

The add-on creates an isolated, versioned environment outside Blender, with its
own downloaded Python 3.11, PyTorch and the dependencies of a pinned Kimodo source
revision. Blender 5.2 keeps using its own bundled Python. Setup automatically
fills in the backend Python/repository paths after verification succeeds.

The managed profile currently supports **Windows x64** and **Linux x86-64**, with
an NVIDIA CUDA 12.8 build or an explicitly selected CPU-only build. A compatible
NVIDIA driver is required for the GPU profile; the installer tests a CUDA tensor
operation and does not silently fall back to CPU or install/change drivers.
Other platforms can still use an existing external environment.

Default Windows location: `%LOCALAPPDATA%\KimodoMotionStudio`. A different local
drive/folder can be selected in the setup dialog. Layout:

```
<chosen folder>/
  runtimes/py311-torch271-r1-windows-cu128/
    python/           # private managed CPython
    venv/             # Kimodo and PyTorch, never Blender site-packages
    source/           # pinned Kimodo checkout without requiring Git
    tools/            # private checksum-verified uv executable
    cache/            # installer/package cache
    ready.json        # written only after verification succeeds
    probe.json        # versions, package inventory and backend checks
    status.json
    setup-*.log
  models/             # default only when no model folder is already selected
  outputs/            # default only when no output folder is already selected
```

Existing model/download/manual paths and output folders are preserved. Setup does
not delete previous environments or models, change system PATH/registry Python
registration, use administrator rights, or install packages into Blender. A
changed install folder creates a different environment; it does not move files.
Re-enabling the add-on reuses a valid completion marker and does not download
anything. **Verify / repair backend** explicitly reruns installation checks using
the download/package cache. It refuses modified source files and unowned runtime
folders rather than overwriting them; select a new installation folder in that
case. Cancelling setup stops the subprocess tree and retains reusable downloads.

## Models are a separate explicit step

After **Local backend ready**, use **Generate > Local model storage** to select a
model and click **Download / resume**, or select a manual checkpoint folder.
The installer does not download model weights or accept gated Hugging Face terms.
Your existing model manager still supports a chosen model/cache folder and local
file checks. Standard LLM2Vec text encoding needs the Llama base and both adapters.
Any required account/license approval must be completed by the user. Generation
and continuation remain local/offline; no cloud inference or manually started
bridge is used. The worker is automatically launched by the add-on when needed.

## Optional compiled cleanup

This clean managed profile deliberately does **not** compile the optional C++
`MotionCorrection` extension or install compiler toolchains. Core generation,
constraints and the add-on's SLERP/root continuation blending remain available.
The **Kimodo foot-skate/constraint cleanup** option must remain off in managed
mode; attempting to use it produces a specific message, not a silent fallback.
Use an advanced external environment with MotionCorrection installed to use that
optional postprocessing path. The original skeleton/retargeting limitations in
`BLENDER_README.md` still apply.

## Integrity, failure reporting and validation

The uv bootstrap is version-pinned and checked against published SHA-256 hashes.
The downloaded Kimodo ZIP is checked against the entire pinned Git tree before
any source is extracted or setup.py runs. Package installation names only the
owned venv and uses binary wheels for third-party dependencies, except the pinned pure-Python
ANTLR 4.9.3 runtime required by Hydra, which ships only as source. The local Kimodo
package is built with its optional C++ extension disabled. The profile pins Python,
PyTorch and Transformers; other dependencies follow the pinned upstream manifest,
so this is not a fully locked transitive dependency environment. Exact installed
versions are recorded in `probe.json`.

Setup reports stages and actual logs, not a fabricated percentage. **Ready** means
imports, built-in skeleton assets and a tensor operation passed. It does not mean
model weights are present or real inference has been tested. Errors/cancellation
never publish a ready marker. Use **Open setup folder / logs** for details.

Local unit/fixture tests run with `python -m pytest blender_tests -q`.
`blender_tools/blender_smoke_test.py` checks real bpy registration and existing
motion/JSON features; headless registration must not start installation.
`blender_tools/managed_install_smoke.py` is an explicit, networked CPU-install
integration test for disposable CI machines. It tests setup and second-run reuse,
without model weights. The CI matrix covers Windows/Linux installation and bpy
4.5.3/5.2.1 registration. Check actual Actions results rather than assuming a
workflow definition proves that installation passed. Interactive first-enable
popup behavior, local NVIDIA drivers, gated authentication and full model inference
need end-to-end validation on the target computer.

References reviewed for this implementation:
- Blender online-access policy: https://docs.blender.org/api/current/bpy.app.html
- uv managed Python: https://docs.astral.sh/uv/guides/install-python/
- uv environment controls: https://docs.astral.sh/uv/reference/environment/
- Verified uv 0.8.22 files/hashes: https://pypi.org/project/uv/0.8.22/
- PyTorch 2.7.1 CPU/CUDA wheels: https://pytorch.org/get-started/previous-versions/
