# main_sonic_test.py
"""
Test Sonic environment.
Runs a random episode, prints total reward,
and saves detailed logs (actions + rewards) as CSV.
"""

from environment_sonic import make_env
from resetstate_sonic import ResetStateWrapper
import pandas as pd
import os

# =============================
# 1️⃣ إنشاء البيئة
# =============================
env = make_env(render=True)

out = env.reset()
if isinstance(out, tuple):
    obs, info = out
else:
    obs, info = out, {}

total_reward = 0.0
done = False

print("🚀 Running random test episode...\n")

# =============================
# 2️⃣ حلقة الأكشنات العشوائية
# =============================
for step_idx in range(1000):  # تجربة قصيرة
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

    if done:
        print(f"🏁 Episode finished at step {step_idx}")
        break

print(f"\n✅ Total reward: {total_reward}")

# =============================
# 3️⃣ البحث عن ResetStateWrapper داخل السلسلة
# =============================
def find_wrapper(env, wrapper_type):
    """Traverse nested wrappers to locate the desired one."""
    current = env
    depth = 0
    while hasattr(current, "env"):
        if isinstance(current, wrapper_type):
            print(f"✅ Found wrapper {wrapper_type.__name__} at depth {depth}")
            return current
        current = current.env
        depth += 1
    return None

rs = find_wrapper(env, ResetStateWrapper)

# =============================
# 4️⃣ حفظ التقرير (حتى لو لم تُنهِ البيئة الحلقة)
# =============================
if rs is not None and getattr(rs, "episode_records", None):
    if rs.episode_records:
        df = pd.DataFrame(rs.episode_records)
        rs.episode_id += 1
        path = os.path.join(rs.log_dir, f"episode_{rs.episode_id:03d}.csv")
        df.to_csv(path, index=False)
        print(f"📄 Episode log saved → {path}")
    else:
        print("⚠️ ResetStateWrapper found but no records collected.")
else:
    print("⚠️ No ResetStateWrapper found — no logs to save.")

# =============================
# 5️⃣ إغلاق البيئة
# =============================
env.close()
print("\n🎮 Test finished successfully!")
