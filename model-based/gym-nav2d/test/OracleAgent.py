import gym
import time
import math
import numpy as np

DEBUG = True

for e in range(100):
    # env = gym.make('gym_nav2d:nav2dVeryEasy-v0')
    # env = gym.make('gym_nav2d:nav2dEasy-v0')
    # env = gym.make('gym_nav2d:nav2dHard-v0')
    env = gym.make('gym_nav2d:nav2dVeryHard-v0')
    obs = env.reset()
    cumulated_reward = 0

    i = 0
    done = False
    while not done and i <= 100:
        i += 1

        agent_x = env.agent_x
        agent_y = env.agent_y

        goal_x = env.goal_x
        goal_y = env.goal_y

        # do some basic geometry with triangles ;-)
        distance = env._distance()
        adjacent = (goal_y - agent_y)
        disjacent = (goal_x - agent_x)
        angle = math.atan2(adjacent, disjacent) + math.pi*1.5

        if distance > 10:
            dist = 10
        else:
            dist = distance

        if angle > 2*math.pi:
            angle -= 2*math.pi

        angle_grad = angle/(2*math.pi)*360
        angle_a = angle/(2*math.pi)*2-1
        dist_a = dist/10*2-1
        act = np.array([angle_a, dist_a])
        obs, rew, done, info = env.step(act)     # take a random action
        env.render(mode='human')

        cumulated_reward += rew
        if DEBUG:
            print(info)
    print("episode ended with cumulated rew", cumulated_reward, "and done:", done)
    if DEBUG and done:
        time.sleep(3)
    env.close()
