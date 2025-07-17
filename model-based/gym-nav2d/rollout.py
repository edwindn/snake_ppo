import matplotlib
matplotlib.use('Agg') # use on Mac
import matplotlib.pyplot as plt
import imageio
from gym_nav2d.envs import Nav2dEnv
import numpy as np


env = Nav2dEnv()

obs = env.reset()
done = False
i = 0
frames = []

while not done and i<100:
    i += 1
    action = env.action_space.sample() # taking random actions
    obs, *_ = env.step(action)
    state = env.render(mode='ansi')
    state = [s/255 for s in state] # normalise to [0, 1]
    
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(state[0], state[1], c='blue', label='agent')
    ax.scatter(state[2], state[3], c='red',  label='goal')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend()
    ax.set_title(f"Step {len(frames)}  dist={state[4]:.2f}")
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    frames.append(image)

env.close()

imageio.mimwrite('rollout.mp4', frames, fps=30)

print(f"Saved video of length ({len(frames)} frames) to rollout.mp4")