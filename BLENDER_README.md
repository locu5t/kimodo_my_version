# Kimodo Motion Studio — native Blender alpha 0.2.0

A native Blender adaptation of Kimodo's authoring workflow, not a browser window
embedded in Blender. The 3D View is the motion viewer, the Dope Sheet contains
colored prompt/constraint tracks, and the Kimodo sidebar groups controls into
**Generate, Constraints, Load / Save, Visualize, and Help**.

Development repository: `locu5t/kimodo_my_version`, branch
`feature/blender-motion-studio`. NVIDIA's original model code and `main` are not
modified by this integration.

## Install

Build with `python blender_tools/build_addon.py`, then install
`blender_dist/kimodo_motion_studio-0.2.0.zip` using Blender's **Install from Disk**.
The manifest is at the archive root. Do not install a GitHub source ZIP as an
extension. Disable/remove the earlier `kimodo_blender` 0.1.0 extension or old
scaffold first, to avoid duplicate `kimodo.*` operators.

Open **N → Kimodo** in the 3D View or Dope Sheet. An optional **Create Kimodo
workspace** button duplicates the current workspace; enlarge its Dope Sheet to
show all seven lanes. This uses Blender's editors, panels, playback and selection.
It does not replace Blender with Viser or change the original Kimodo web UI.

In **Load / Save → External backend**, select the Python executable from a
working Kimodo environment and this repository's local path. PyTorch runs in an
external worker, not inside Blender. Native Windows/Linux and WSL command
construction is included. WSL workers and job files must be on Windows drives
mapped under `/mnt/<drive>`; UNC paths are not supported. Check the backend before
generation. Model downloads are disabled unless explicitly permitted.

## Interface and features

| Section | Implemented path |
| --- | --- |
| Generate | Rigplay/SEED, SOMA/G1/SMPL-X model selectors, versions/custom model ID, native rig preparation, prompts, seed, steps, guidance, heading, transition and cleanup controls |
| Prompt timeline | Double-click text editing, right-edge ripple trimming, drag-body reordering, append on empty row, right-click menu, mute and delete |
| Constraints | Separate Full Body, 2D Root, Left/Right Hand and Left/Right Foot lanes; single-frame or sampled-interval capture; move/retime/mute/copy/delete |
| Pose editing | Temporary native editing rig; apply/cancel changes without overwriting the source Action; a pose edit applies one pose across the selected interval |
| Paths | Movable native Empty root waypoints and selected POLY/Bezier curves sampled to 2D paths; curve changes require recapture |
| JSON | Prompt arrays, native meta.json, native constraints.json, and Studio project JSON roundtrips |
| Load/Save | Native SOMA BVH, Kimodo/AMASS NPZ, G1 CSV import; example meta/constraints/motion loading; source export to NPZ/transfer NPZ and optional SOMA BVH |
| Continue | Actual source-tail motion context, extra conditioning frames, world alignment, optional quaternion/root blending, new combined armature/Action |
| Variations | 1–8 takes, separate reproducible seeds, select/solo/show all/use as source; model cached across samples within one worker job |
| Visualize | Skeleton style, bone names, in-front drawing, grid/shading, frame source, sampled root trajectory, track visibility |

The generate buttons use the same public Kimodo inference/constraint API as the
original demo. Controls are wired into job requests, not placeholders. The
**Load model / create native rig** action validates model loading and creates a
neutral rig; it does not keep the model resident after that worker exits. A later
generation starts a new worker. No fabricated inference progress percentage is
shown; stage status and worker logs are available.

## Timed prompt JSON

```json
[
  {"text": "A person walks with a strut", "duration": 6},
  {"text": "The person turns around confidently", "duration": 3},
  {"text": "The person stops and waves", "duration": 2.5}
]
```

Also accepted: `{"texts":[...],"durations":[...]}`, a single native
`{"text":"...","duration":6}`, or `{"prompts":[...]}`. Per-entry
`start`/`start_seconds` and `duration`/`duration_seconds` are recognized. Explicit
starts are relative to the generated section, must begin at zero and be
contiguous. Durations can be numeric strings, such as `"6"`.

Moving a prompt reorders a **sequential** sequence. Trimming its right edge ripples
later prompts; it does not create gaps or simultaneous prompt layers. Seconds
are authoritative. Cumulative rounding avoids timing drift; scene FPS and model
FPS are separate. Non-finite values, empty text, overlaps, gaps, sub-frame
segments and oversized files are rejected. Start with short 2–10-second prompts;
parser safety ceilings are not model-capacity guarantees.

**Studio project JSON** retains prompt enable states, settings and captured
constraints. It deliberately does not contain executable backend paths, download
permissions, model weights or animation Actions. Save the `.blend` for the rigs
and Actions. Loading requires matching scene FPS, rather than silently retiming
an existing scene. Native constraints JSON does not specify its FPS: set
**Native constraints JSON FPS** correctly when importing/exporting it.

## Constraints

First import/generate a native Kimodo rig, or use **Load model / create native
rig**. Choose it as **Source rig**. Click a constraint lane to capture that pose,
or Ctrl-drag an empty lane to capture the motion over an interval. Drag items to
move them; interval-edge dragging retimes their sampled data. Escape restores a
cancelled drag. Mouse wheel/MMB retain Blender's navigation.

Use **Edit pose / waypoint** for a full-body or hand/foot constraint. The temporary
rig is separate from the source. Change bone rotations in Pose Mode, then
**Apply edit** or **Cancel edit**. This alpha's pose editor turns an interval into
a held edited pose; use **Recapture** to retain a moving interval. Bone stretching,
renamed hierarchies, changed proportions and non-rigid scaling are rejected.

For root waypoints, enable **Root waypoint from 3D cursor** before capture. Move
the resulting Empty with Blender's normal transform tools. For a path, select a
single POLY/Bezier curve, set the interval, then use **Selected curve to 2D root
path**. Paths are sampled by arc length; Blender Z height is ignored for 2D roots.
Intervals are inclusive; prompt spans are half-open. The compiler converts each
once to model frames, offsets for continuation context, and rejects out-of-range
or same-track overlapping constraints rather than silently dropping them.

Fresh generation can align its origin/heading to a frame-zero body constraint or
root waypoint. Continuation always uses the source context's frame. Constraint
following and smooth velocity at a seam remain model-dependent, not guaranteed.

## Continue an imported animation

Choose the source rig and **Use active Action range**. Import the desired prompt
JSON, enable the appended preview, and use **Continue from end**. The source is
sampled with its evaluated Action/NLA, constraints and rigid object transform.
Its actual final moving frames condition a newly generated suffix.

Context is extra: at 30 FPS, 180 source samples plus a six-second prompt produce
360 output samples. The overlapping conditioning samples are removed exactly
once. Blend 0 preserves source samples at matching FPS; a nonzero blend affects
only the tail in the new combined result. The original Action remains unchanged.
Skeleton joint order, rest orientation and proportions must match the model.
This is not automatic FBX/Mixamo/Rigify retargeting.

## Validation and limits

See `BLENDER_VALIDATION.json` for the local build results. Pure timing/math,
constraint-compilation and mocked-model contract tests are runnable with:

```powershell
python -m pip install numpy pytest
python -m pytest blender_tests -q
```

Torch is optional for core-only tests; four original plus three new mock-model
contract tests need it. The supplied real-bpy smoke script checks registration,
rig baking/roundtrip, cancelled-bake rollback, JSON import, constraint capture and
pose editing, and project persistence. CI defines Blender 4.5.3/Python 3.11 and
Blender 5.2.1/Python 3.13 smoke jobs. A workflow definition is not proof that those
jobs ran or passed. Check the actual Actions results.

**Development alpha, not a complete feature-for-feature port.** Interactive mouse
handling and GPU drawing, real-model inference, CUDA/Windows/WSL behavior and
visual seam quality need end-to-end validation. Automatic skinned SOMA/SMPL-X
meshes, persistent model-server sessions, IK target gizmos for individual
end-effectors, arbitrary-rig retargeting, interior motion inpainting, audio/beat
conditioning and mixed-character timelines are not implemented. Import formats
come from Kimodo; export in this alpha is NPZ and SOMA BVH, not every web-demo
export format. Root-path snapshots and orphaned waypoint objects are retained
rather than silently deleted.

## Source and licensing

Reviewed upstream API: `nv-tlabs/kimodo` at
`1aece8c124d73d255ceff5086d983b844c9f4e94`, which is this fork's inherited base.
The earlier local prototype was compared against `Aero-Ex/kimodo`; this version
uses only the shared public APIs, not its fork-only per-segment seed argument.

References:
- https://research.nvidia.com/labs/sil/projects/kimodo/docs/interactive_demo/ui_overview.html
- https://research.nvidia.com/labs/sil/projects/kimodo/docs/api_reference/model.html
- `kimodo/constraints.py`, `kimodo/exports/motion_io.py`, `kimodo/model/registry.py`

The new add-on is GPL-3.0-or-later; its LICENSE/NOTICE are included. Kimodo code,
model weights, body assets and third-party components retain their existing
licenses. No model weights or protected body-model assets are bundled.
