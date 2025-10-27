import time
import numpy as np
import pygame
import gymnasium as gym
import retro

# --- Create env with on-screen rendering (human) ---
env = retro.make(game="SonicTheHedgehog2-Genesis", state="EmeraldHillZone.Act1", render_mode="human")
obs, info = env.reset()

# Buttons order from retro (for Genesis):
# Typically: ['B','C','A','START','UP','DOWN','LEFT','RIGHT']
buttons = env.buttons
btn_idx = {name: i for i, name in enumerate(buttons)}

# Map keyboard -> console buttons
KEYMAP = {
    pygame.K_RIGHT: "RIGHT",
    pygame.K_LEFT:  "LEFT",
    pygame.K_UP:    "UP",
    pygame.K_DOWN:  "DOWN",
    pygame.K_z:     "A",
    pygame.K_x:     "B",
    pygame.K_c:     "C",
    pygame.K_RETURN:"START",
}

pygame.init()
screen = pygame.display.set_mode((400, 50))
pygame.display.set_caption("Sonic manual play — use arrows + Z/X/C, press S to save combos")

pressed = set()
combo_counts = {}

clock = pygame.time.Clock()
running = True
step_hz = 60  # target steps per second

def action_from_pressed():
    act = np.zeros(len(buttons), dtype=np.int8)
    for name in pressed:
        if name in btn_idx:
            act[btn_idx[name]] = 1
    return act

def combo_str():
    if not pressed: return "NO-OP"
    return "+".join(sorted(pressed))

last_print = 0

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in KEYMAP:
                pressed.add(KEYMAP[event.key])
            elif event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_s:
                # Save discovered combos
                with open("discovered_combos.txt", "w") as f:
                    for k, v in sorted(combo_counts.items(), key=lambda x: -x[1]):
                        f.write(f"{k}\t{v}\n")
                print("💾 Saved combos -> discovered_combos.txt")
        elif event.type == pygame.KEYUP:
            if event.key in KEYMAP and KEYMAP[event.key] in pressed:
                pressed.remove(KEYMAP[event.key])

    # Build action & step
    act = action_from_pressed()
    obs, reward, terminated, truncated, info = env.step(act)
    done = terminated or truncated

    # Track combo frequency
    cs = combo_str()
    combo_counts[cs] = combo_counts.get(cs, 0) + 1

    # Status bar
    now = time.time()
    if now - last_print > 0.25:  # refresh 4x/sec
        txt = f"Combo: {cs}   |  Top: {sorted(combo_counts.items(), key=lambda x:-x[1])[:3]}"
        screen.fill((20,20,20))
        pygame.display.set_caption(txt)
        last_print = now

    if done:
        obs, info = env.reset()

    clock.tick(step_hz)

env.close()
pygame.quit()

# Auto-save on exit
with open("discovered_combos.txt", "w") as f:
    for k, v in sorted(combo_counts.items(), key=lambda x: -x[1]):
        f.write(f"{k}\t{v}\n")
print("✅ Done. Combos saved to discovered_combos.txt")
