🚀 Add full PPO training loop with entropy scheduling and checkpoint auto-resume

- Implemented full PPO update (policy, value, entropy losses) with GAE advantage computation.
- Added clipped policy and value loss (standard PPO objective).
- Introduced entropy coefficient schedule (0.05 → 0.01 over 200k steps) to improve exploration early on.
- Normalized advantages and applied gradient clipping for stability.
- Fixed env.reset() logic to avoid double resets and ensure info handling.
- Improved checkpointing:
  - Auto-loads latest checkpoint (sonic_ppo_latest.pt or highest *_k.pt)
  - Saves both incremental and "latest" model snapshots
- Added progress heartbeat logs every 200 steps for long training runs.
- Made RecordVideo trigger only on save intervals to avoid performance slowdown.
- Compatible with both new and old Gym/Gymnasium step APIs.
- Now the model *actually learns* instead of only collecting rollouts.

