import matplotlib
matplotlib.use('Agg') # use on Mac
import matplotlib.pyplot as plt
import imageio
from nav_env import NavEnv
import numpy as np


env = NavEnv()

debug = False

obs, _ = env.reset()
done = False
i = 0
frames = []

while not done and i<100:
    i += 1
    action = env.random_action()

    if debug:
        print(f"action: {action}")

    obs, *_ = env.step(action)

    if debug:
        print(f"obs: {obs}")
        
    state = obs.tolist()[:-1]

    if debug:
        print(f"state: {state}")

    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(state[0], state[1], c='blue', label='agent')
    ax.scatter(state[2], state[3], c='red',  label='goal')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend()
    ax.set_title(f"Step {len(frames)}  dist={obs[4]:.2f}")
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
    if debug:
        print(f"Array size: {image.size}, Expected size: {400 * 400 * 3}")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]

    plt.close(fig)

    frames.append(image)

env.close()

imageio.mimwrite('rollout.mp4', frames, fps=30)

print(f"Saved video of length ({len(frames)} frames) to rollout.mp4")