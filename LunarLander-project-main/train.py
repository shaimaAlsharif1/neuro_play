

#import and set up
from __future__ import annotations
import random
import numpy as np
import torch

#import from other files in the dirc
from agent import DQNAgent
from config import DQNConfig
from environment import make_env
from gymnasium.wrappers import RecordVideo
import os





def train_dqn(train_steps: int = 1000, eval_episodes: int = 5, render_eval: bool = True, seed: int = 1):

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
     #setting seeds to ensures that results are reproducible
    env = make_env(render=False)
    #make the env
    obs_dim = env.observation_space.shape[0]
    #size of state vector, position, velocity, angle, leg contacts
    act_dim = env.action_space.n
    #nu,ber of discrete actions, in our case is 4: do nothing, left, main, right
    device = "cuda" if torch.cuda.is_available() else "cpu" # chose the gpu
    # agent and configrations
    cfg = DQNConfig()
    agent = DQNAgent(obs_dim, act_dim, device, cfg)

    s, _ = env.reset(seed=seed)
    #reset the rnv to start the training, s is the initial state
    episode_return = 0.0
    # track the total reward per episode

    #training loop:
    for t in range(1, train_steps + 1): #loops over the train steps
        a = agent.act(s)
        # the agent select an action

        # agent perform the action
        ns, r, done, tr, _ = env.step(a)
        agent.push(s, a, r, ns, float(done or tr))
        #store the experience, save the output of the previous step

        # calculate the loss
        loss = agent.train_step()

        # move to the next state
        s = ns
        episode_return += r
        # if the agent is stopped or turnecated : save the result and reset the env
        if done or tr:
            print(f"Step {t:6d} | eps={agent.epsilon():.3f} | return={episode_return:.1f} | replay={len(agent.replay)}" )
            episode_return = 0.0
            s, _ = env.reset()
    env.close()

    torch.save(agent.q.state_dict(), "dqn_lunarlander.pth")
    # save the file

    # Evaluate
    print("\nEvaluating...")
    #create a new env for evaluation
    eval_env = make_env(render_eval)
    try:
        returns = []
        #loops over the episodes
        for ep in range(1, eval_episodes + 1):
            s, _ = eval_env.reset()
            done, tr = False, False
            total = 0.0
            while not (done or tr):
                with torch.no_grad():
                    q = agent.q(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
                    a = int(torch.argmax(q, dim=1).item())
                s, r, done, tr, _ = eval_env.step(a)
                if render_eval:
                    eval_env.render(mode="human")
                total += r
            returns.append(total)
            print(f"[Eval] Episode {ep}/{eval_episodes} | Return: {total:.2f}")
        print(f"Avg return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    finally:
        eval_env.close()
