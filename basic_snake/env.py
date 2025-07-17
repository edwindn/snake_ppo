import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
import numpy as np
import cv2

class Environment(gym.Env):
    """
    Implements a Snake game environment for RL training.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, **config):
        super().__init__()
        self.debug = config.get('debug', False)
        self.size = config.get('grid_size')
        self.initial_length = config.get('initial_length', 3)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: straight, 1: right, 2: left
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.size, self.size), dtype=np.uint8) # replace with image later
        
        # Direction map: 0:right, 1:down, 2:left, 3:up
        self._direction_map = {
            0: np.array([1, 0]),   # right
            1: np.array([0, -1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1])    # up
        }
        
        self.dist_rew_coef = 0.1 # for distance reward

        self.render_mode = config.get('render_mode', 'rgb_array')
        self.cell_size = 20  # Pixels per grid cell for rendering
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        self.frame_count = 0
        self.score = 0
        
        # Initialize snake
        initial_x = np.random.randint(self.initial_length - 1, self.size)
        initial_y = np.random.randint(0, self.size)
        self._head_location = np.array([initial_x, initial_y])
        body_locations = [[initial_x - i, initial_y] for i in range(1, self.initial_length)]
        self._body_location = np.array(body_locations)
        self._snake = np.vstack([self._head_location, self._body_location])
        
        # if self.debug:
        #     print(f"Snake shape: {self._snake.shape}")
        #     print(f"Initial snake position: {self._snake}")
        
        self._current_direction = 0  # Start facing right
        self._place_target()
        
        return self._get_obs(), {'score': self.score}

    def _place_target(self):
        """Place the target at a random position not occupied by the snake."""
        snake_positions = self._snake.tolist()
        while True:
            target_x = np.random.randint(0, self.size)
            target_y = np.random.randint(0, self.size)
            if [target_x, target_y] not in snake_positions:
                break
        self._target_location = np.array([target_x, target_y])
        # if self.debug:
        #     print(f"Target position: {self._target_location}")

    def step(self, action):
        """
        Update the environment based on the action.
        Actions: 0 (straight), 1 (right), 2 (left)
        """
        assert action in [0, 1, 2], f"Action must be 0, 1, or 2, but got {action}"
        self.frame_count += 1

        distance = np.sum(np.abs(self._snake[:, :] - self._target_location))
        
        # Update direction: 0=straight, 1=right(+1), 2=left(-1)
        delta_direction = [0, 1, -1][action]
        self._current_direction = (self._current_direction + delta_direction) % 4
        
        # Calculate new head position
        head_coords = self._snake[0, :] + self._direction_map[self._current_direction]

        new_distance = np.sum(np.abs(head_coords - self._target_location))
        distance_reward = (distance - new_distance) * self.dist_rew_coef
        
        terminated = False
        truncated = False
        reward = 0.0
        
        # Check for collision or time limit
        if self._is_collision(head_coords):
            terminated = True
            reward = -1.0
        elif self.frame_count > 50 * self._snake.shape[0]:
            truncated = True
            reward = -1.0 if self.score==0 else 0.0

        else:
            # Check if target is eaten
            if np.all(head_coords == self._target_location):
                reward = 10.0
                if self.debug:
                    print('Reward hit!')
                self.score += 1
                self._snake = np.vstack([head_coords, self._snake])
                self._place_target()
            else:
                self._snake = np.vstack([head_coords, self._snake[:-1, :]])
                reward = distance_reward
        
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {'score': self.score}

    def _is_collision(self, head_coords):
        """Check if the head collides with the body or walls."""
        if (head_coords[0] < 0 or head_coords[0] >= self.size or 
            head_coords[1] < 0 or head_coords[1] >= self.size):
            return True
        if any(np.all(head_coords == self._snake[i, :]) for i in range(1, self._snake.shape[0])):
            return True
        return False

    def _get_obs(self):
        """Return the current grid state as an observation."""
        return self._get_grid()

    def _get_grid(self):
        """Generate a 2D grid representing the current state."""
        grid = np.zeros((self.size, self.size), dtype=np.uint8)
        for i, pos in enumerate(self._snake):
            x, y = pos
            grid[x, y] = 2 if i == 0 else 1  # 2 for head, 1 for body
        tx, ty = self._target_location
        grid[tx, ty] = 3  # 3 for target
        return grid

    def render(self, mode='rgb_array'):
        """Render the current state as an RGB array."""
        if mode == 'rgb_array':
            grid = self._get_grid()
            img = np.zeros((self.size * self.cell_size, self.size * self.cell_size, 3), dtype=np.uint8)
            # Colors: 0=empty(white), 1=body(green), 2=head(dark green), 3=target(red)
            colors = {
                0: (255, 255, 255),  # White
                1: (0, 255, 0),      # Green
                2: (0, 100, 0),      # Dark Green
                3: (255, 0, 0)       # Red
            }
            for i in range(self.size):
                for j in range(self.size):
                    color = colors[grid[i, j]]
                    img[i*self.cell_size:(i+1)*self.cell_size, 
                        j*self.cell_size:(j+1)*self.cell_size, :] = color
            return img
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")



if __name__ == '__main__':
    config = {
        'grid_size': 20,
        'initial_length': 3,
        'debug': True,
        'render_mode': 'rgb_array'
    }

    env = Environment(**config)
    env = RecordVideo(env, video_folder='snake_videos', name_prefix='test_rollout')

    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if config['debug']:
            print(f"Action: {action}, Reward: {reward}, Score: {info['score']}, Done: {done}")

    env.close()


    # env = gym.make('CartPole-v1')
    # env = gym.wrappers.RecordVideo(env, "videos")

    # obs = env.reset()
    # tot_reward = 0

    # for _ in range(200):
    #     action = env.action_space.sample()
    #     # out = env.step(action)
    #     # print(out)
    #     # exit()
    #     obs, reward, done, info = env.step(action)
    #     tot_reward += reward
    #     if done:
    #         obs = env.reset()
    #         print(f'Total return: {tot_reward}')
    #         tot_reward = 0
    
    # env.close()
