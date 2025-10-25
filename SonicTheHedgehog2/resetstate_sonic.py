# resetstate_sonic.py
import gymnasium as gym
import numpy as np
import os
import pandas as pd

class ResetStateWrapper(gym.Wrapper):
    """
    Custom reward shaping + action logging for Sonic the Hedgehog 2.
    - Encourages forward progress
    - Penalizes idling, ring loss, and useless jumps
    - Logs each step (action, reward, progress) to CSV after every episode
    """

    def __init__(self, env, max_steps=4500, log_dir="logs"):
        super().__init__(env)
        self.env = env
        self.max_steps = max_steps
        self.steps = 0
        self.prev_info = None
        self.jump_counter = 0

        # 🟢 Logging setup
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_records = []
        self.episode_id = 0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}
        self.steps = 0
        self.prev_info = info
        self.jump_counter = 0
        self.episode_records = []
        return obs, info

    def step(self, action):
        """Performs one environment step with custom rewards and logs."""

        step = self.env.step(action)

        # ✅ دعم كلتا الصيغتين (Gym/Gymnasium)
        if len(step) == 5:
            obs, reward, terminated, truncated, info = step
            done = terminated or truncated
        else:
            obs, reward, done, info = step
            terminated, truncated = done, False

        # -----------------------------
        # 🧮 حساب المكافأة المخصصة
        # -----------------------------
        custom_reward = 0.0

        # معلومات اللعبة
        x = info.get("x", 0)
        rings = info.get("rings", 0)
        lives = info.get("lives", 3)
        score = info.get("score", 0)
        screen_x_end = info.get("screen_x_end", 10000)

        if self.prev_info is None:
            self.prev_info = info

        prev_x = self.prev_info.get("x", 0)
        prev_rings = self.prev_info.get("rings", 0)
        prev_lives = self.prev_info.get("lives", 3)

        # 1️⃣ مكافأة التقدم للأمام
        dx = x - prev_x
        if dx > 0:
            custom_reward += 0.1 * (dx / 100.0)
        elif dx == 0:
            custom_reward -= 0.01  # عقوبة التوقف

        # 2️⃣ عقوبة فقدان خواتم
        if rings < prev_rings:
            custom_reward -= 0.3

        # 3️⃣ مكافأة الاقتراب من نهاية المرحلة
        custom_reward += (x / screen_x_end) * 0.5

        # 4️⃣ عقوبة فقدان حياة
        if lives < prev_lives:
            custom_reward -= 1.0
            done = True

        # 5️⃣ مكافأة الوصول للنهاية
        if x >= screen_x_end:
            custom_reward += 1.0
            done = True

        # -----------------------------
        # 🎯 منطق القفز الذكي
        # -----------------------------
        buttons = getattr(self.env.unwrapped, "buttons", [])
        jump_buttons = ['A', 'B', 'C']

        # إذا الأكشن رقم (Discrete) نحوله إلى مصفوفة أزرار
        if hasattr(self.env, "action") and isinstance(action, (int, np.integer)):
            try:
                action_array = self.env.action(action)
            except Exception:
                action_array = np.zeros(len(buttons), dtype=np.int8)
        else:
            action_array = action

        # تحديد الأزرار المضغوطة
        pressed_buttons = [buttons[i] for i, val in enumerate(action_array) if val == 1]
        is_jump = any(b in pressed_buttons for b in jump_buttons)

        # تتبع عدد القفزات المتتالية
        if is_jump:
            self.jump_counter += 1
        else:
            self.jump_counter = 0

        # عقوبة القفز في نفس المكان
        if is_jump and dx <= 0:
            custom_reward -= 0.02

        # عقوبة القفز المتكرر (spam)
        if self.jump_counter > 3:
            custom_reward -= 0.1 * (self.jump_counter - 3)

        # -----------------------------
        # ⏱️ حد أقصى للخطوات
        # -----------------------------
        self.steps += 1
        if self.steps > self.max_steps:
            done = True

        # -----------------------------
        # ⚖️ تطبيع المكافأة
        # -----------------------------
        custom_reward = np.clip(custom_reward, -1.0, 1.0)

        # -----------------------------
        # 🧾 نظام التتبع (Logging)
        # -----------------------------
        action_id = int(action) if isinstance(action, (int, np.integer)) else -1
        record = {
            "step": self.steps,
            "action": action_id,
            "is_jump": bool(is_jump),
            "x": x,
            "rings": rings,
            "reward": round(float(custom_reward), 4),
        }
        self.episode_records.append(record)

        # حفظ التقرير عند نهاية الحلقة
        # Force-save logs if max steps reached (even if not "done")
        if self.steps >= self.max_steps - 1 and self.episode_records:
            df = pd.DataFrame(self.episode_records)
            self.episode_id += 1
            path = os.path.join(self.log_dir, f"episode_{self.episode_id:03d}.csv")
            df.to_csv(path, index=False)
            print(f"📄 Episode log (forced save) → {path}")
            self.episode_records = []

        # تحديث الحالة السابقة
        self.prev_info = info

        # إرجاع القيم بالصيغـة الحديثة (5 قيم)
        terminated = done
        truncated = False
        return obs, custom_reward, terminated, truncated, info
