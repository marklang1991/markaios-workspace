# Blueprint-Bound Keyframe and Microclip Method

## Non-negotiable authority

The matching blueprint frame or clip owns camera, FOV, shoulder, horizon, shot order, hard cuts, actor count and slots, pose, root, crop, occlusion, weapon line, events, topology, timing, and result.

Approved identity, weapon, and scene assets may replace only their assigned domains. They cannot replace the blueprint's screen geometry.

## Keyframe construction

1. Start from the matching source frame as an immutable carrier.
2. Freeze static keylines, actor and weapon masks, roots, crops, occlusion order, and event overlays.
3. Reuse only V10 components with an explicit `PASS`/`FROZEN` state and SHA-256. Never inherit a `PROCESS`, `BLOCK`, `REJECT`, or `NEGATIVE_ONLY` full frame.
4. Project approved scene RGB into source surface masks. Donor edges, openings, occluders, camera, and topology have zero authority.
5. Composite actors and weapons as isolated RGBA or allowed rigid sub-layers inside their source slots. Lock hands and weapons as one unit.
6. Restore source keylines and event overlays at their exact coordinates.
7. Apply only low-frequency color and lighting integration. Do not sharpen or regenerate the full frame.

Promotion requires source/topology, semantics/physics, style/LookDev, continuity, and user visual approval. Static anchor error at native 640 must be no greater than 2 pixels, with 0 pixels as the target.

## Microclip

A motion probe is blocked until at least two adjacent, same-shot, flattened final control frames are approved. The only model inputs are:

- one explicit `first_frame`;
- one explicit `last_frame`;
- the matching single-shot blueprint microclip;
- explicit `--generate-audio=false`.

Loose identity, scene, weapon, or additional anchor images are forbidden.

The current Seedance 2.5 endpoint advertises 4–30 second output at 24 fps; it does not natively support 2–3 seconds. A future 4-second probe may yield a 2–3 second review trim only if every retained frame passes the blueprint gates. It must not be time-stretched or presented as native 3-second generation.

If the probe cannot obey the approved endpoints and blueprint motion simultaneously, the model route stops and the approved static keyframes remain the sole authority.
