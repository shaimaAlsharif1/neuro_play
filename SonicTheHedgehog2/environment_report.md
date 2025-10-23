# Sonic 2 Environment Report

**Generated:** 2025-10-23T08:21:09.476716Z

## Environment
- **Game ID:** `SonicTheHedgehog2-Genesis`
- **State:** `EmeraldHillZone.Act1`

## Observation Space
- **Shape:** `(224, 320, 3)`
- **Dtype:** `uint8`

## Action Space
- **Spec:** `MultiBinary(12)`
- **Buttons (Retro order):**
  - 0: B
  - 1: A
  - 2: MODE
  - 3: START
  - 4: UP
  - 5: DOWN
  - 6: LEFT
  - 7: RIGHT
  - 8: C
  - 9: Y
  - 10: X
  - 11: Z

## Episode Probe
- **Frames saved:** 200
- **Total reward (sample rollout):** 300.00

## Notes
- Frames are saved under `frames/` as `frame_00000.png`, `frame_00005.png`, …
- Full button list is in `actions_list.txt`.
- Rollout used a simple bias-right heuristic for sanity-check, not a trained agent.
