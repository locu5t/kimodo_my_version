# Kimodo Motion Studio for Blender

Early Blender integration for the Kimodo motion diffusion project.

## Current alpha features

- Import timed text-to-motion prompt JSON.
- Accept array format, `{ "prompts": [...] }`, and Kimodo-native `texts` / `durations` metadata.
- Edit prompts and durations in Blender.
- Build Blender timeline markers from prompt durations using the active scene FPS.
- Configure a source armature, continuation context frames, blend frames, Kimodo repository path, Python environment, model name, and output path.

## Planned continuation pipeline

1. Sample the tail of the selected Blender animation.
2. Convert it into Kimodo-compatible motion context/constraints.
3. Generate the imported timed prompt sequence with `multi_prompt=True`.
4. Anchor the generated continuation to the source animation ending pose/root heading.
5. Blend the join using root interpolation and quaternion SLERP.
6. Bake the combined result to a new Blender Action without replacing the source Action.

The implementation is intentionally being developed on a feature branch before merging into `main`.

## Example prompt JSON

```json
[
  {"text": "A person walks with a strut", "duration": 6.0},
  {"text": "The person turns around confidently", "duration": 3.0},
  {"text": "The person stops and waves", "duration": 2.5}
]
```

## Install during development

Add the `blender_addon/kimodo_motion_studio` directory as a Blender add-on package, or zip that directory and install it through Blender's add-on installer.
