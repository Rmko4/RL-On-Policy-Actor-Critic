import gymnasium as gym
import time

# Set the environment ID
env_id = 'Ant-v4'
num_envs = 2
render_mode = None
# render_mode = 'human'


def run(envs: gym.vector.VectorEnv):
    episode = 0
    # Reset the environment
    obs = envs.reset()

    # Run a few episodes
    for step in range(5000):
        done = num_envs * [False]
        truncated = num_envs * [False]
        total_reward = num_envs * [0]

        time.sleep(0.1)
        # Choose a random action
        action = envs.action_space.sample()

        # Perform the action in the environment
        obs, reward, done, truncated, info = envs.step(action)

        # Update the total reward
        total_reward += reward

        # Note that env will auto reset when done is True.
        for i in range(num_envs):
            if done[i] or truncated[i]:
                print("Episode:", episode, "Total Reward:", total_reward[i])
                episode += 1
                total_reward[i] = 0


    # Close the environment
    envs.close()


if __name__ == "__main__":
    # Create the environment
    envs = gym.vector.make(env_id, num_envs=num_envs, asynchronous=True,
                           max_episode_steps=10, render_mode=render_mode)
    pass
    run(envs)
    pass

# envs = gym.vector.AsyncVectorEnv(
#     [lambda: gym.make(env_id, render_mode='human') for _ in range(4)])
# env = gym.make(env_id)
