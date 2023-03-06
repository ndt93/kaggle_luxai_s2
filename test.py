import string
import random

import cv2
import numpy as np
from luxai_s2.env import LuxAI_S2
from agent import Agent


def animate(imgs):
    video_name = ''.join(random.choice(string.ascii_letters) for i in range(18)) + '.webm'
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)
    video.release()


def interact(env, agents, steps):
    # reset our env
    obs = env.reset(seed=41)
    np.random.seed(0)
    imgs = []
    step = 0

    while env.state.real_env_steps < 0:
        if step >= steps:
            break
        actions = {}
        for player in env.agents:
            o = obs[player]
            if step == 0:
                a = agents[player].bid_policy(step, o)
            else:
                a = agents[player].factory_placement_policy(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        imgs += [env.render("rgb_array", width=640, height=640)]

    done = False
    while not done:
        if step >= steps:
            break
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].act(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        imgs += [env.render("rgb_array", width=640, height=640)]
        done = dones["player_0"] and dones["player_1"]

    animate(imgs)


if __name__ == '__main__':
    env = LuxAI_S2()
    obs = env.reset(seed=42)
    agents = {player: Agent(player, env.state.env_cfg) for player in env.agents}
    interact(env, agents, 500)
