import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from collections import deque
import random
import time
import yaml

# simulates the agent acting in env, yielding every N steps
# (decouples episode reseting mechanics from the training alg)
def experience_generator(agent, env, N):
    s = env.reset()
    n_steps = 0
    n_eps = 0
    last_cum_rew = 0
    cum_rew = 0
    while True:
        n_steps += 1
        a = agent.pi(s)
        sp, r, done,_ = env.step(a)
        cum_rew += r
        if done:
            n_eps += 1
            last_cum_rew = cum_rew
            cum_rew = 0
            s = env.reset()
        else:
            agent.store_experience(s, a, r, sp, done)
            s = sp

        if n_steps % N == 0:
            yield (n_steps, n_eps, last_cum_rew)



def train_agent(agent, env,
                max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0, # time constraint
                n_transitions_between_updates=100,
                n_optim_steps_per_update=100,
                n_iters_per_p_update=100,
                ):

    # run an episode, and feed data to model
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    exp_gen = experience_generator(agent, env, n_transitions_between_updates)

    while True:
        iters_so_far += 1
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        print("********** Iteration %i ************"%iters_so_far)

        # gather experience
        timesteps_so_far, episodes_so_far, last_cum_rew = exp_gen.__next__()

        # optimize the model from collected data:
        for i in range(n_optim_steps_per_update):
            agent.update_model()

        if iters_so_far % n_iters_per_p_update == 0:
            agent.update_P()

        print("\tLast Episode Reward: %d"%last_cum_rew)
        # add other logging stuff here
        # add saving checkpoints here
