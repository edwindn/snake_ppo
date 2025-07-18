from ray.rllib.algorithms.dreamerv3 import DreamerV3Config

# Configure the algorithm
config = (
    DreamerV3Config()
    .environment("your_navigation_env")  # Replace with your environment name or instance
    # Add other settings, e.g., .training(batch_size=16)
)

algo = config.build()
algo.train()