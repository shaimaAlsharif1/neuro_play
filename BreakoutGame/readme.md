# 🧠 Breakout with Deep Q-Network (DQN)

## 🎮 Project Overview
This project implements a **Deep Q-Network (DQN)** to train an agent to play the **Breakout** Atari game.
The agent learns to **control the paddle** and **break all the bricks** by maximizing its score through **trial and error**.
It uses **Reinforcement Learning**, where the agent interacts with the environment, receives rewards, and improves its decisions over time.

---

### Tools used
- **Python 3.x**
- **TensorFlow / Keras**
- **Gymnasium (Atari environments)**
- **ALE-py**
- **NumPy**
- **Matplotlib** (for training visualization)

---

## How the agent learns:
1. **State Representation:**
   The game frames are preprocessed and stacked to represent the current state of the environment.

2. **Action Selection:**
   The agent uses an **ε-greedy policy** — it explores random actions at first, then gradually exploits what it has learned.

3. **Experience Replay:**
   Past experiences `(state, action, reward, next_state, done)` are stored in a replay buffer and sampled randomly to break correlation and stabilize learning.

4. **Target Network:**
   A secondary network is updated periodically to reduce oscillations during training.

5. **Training:**
   The model minimizes the **Mean Squared Error (MSE)** between predicted and target Q-values.
