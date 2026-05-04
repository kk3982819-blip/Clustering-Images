# Sunrise/Sunset Ground Lighting

## Goal

Make sunrise and sunset floor lighting look like a warm low-angle sun patch entering through the opening:

- bright golden pool immediately below sliding doors/windows
- softer warm spill extending onto the floor
- visible railing/mullion shadow gaps without copying exterior landscape colors
- tile texture preserved under the added light
- effect limited to sunrise and sunset

## Implementation Notes

- Keep sky replacement and room relighting separate.
- Generate the projected floor light from opening geometry plus a 1D opening column profile.
- Use the profile only to create believable light/shadow bands, not to transfer full outdoor texture.
- Add a short high-intensity contact patch near the door threshold and a longer softer floor wash.
- Apply stronger warm lift for sunset than sunrise, but cap the projection to avoid an orange flood.

## Status

Implemented in `full_scene_generator.py`.
