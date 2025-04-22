import math
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

ENV_NAME = "CartPole-v1"
VIDEO_PATH = "videos_qtable"
MODEL_PATH = "qtable_model.npy"
PLOT_PATH = "qtable_reward_plot.png"
N_EPISODES = 300

# 離散化設定
N_BUCKETS = (1, 1, 6, 3)
STATE_BOUNDS = [
    [-2.4, 2.4],
    [-3.0, 3.0],
    [-0.5, 0.5],
    [-math.radians(50), math.radians(50)]
]

def get_state(obs):
    state = []
    for i in range(len(obs)):
        low, high = STATE_BOUNDS[i]
        val = obs[i]
        if val <= low:
            bucket = 0
        elif val >= high:
            bucket = N_BUCKETS[i] - 1
        else:
            scale = (val - low) / (high - low)
            bucket = int(scale * N_BUCKETS[i])
        state.append(bucket)
    return tuple(state)

def get_epsilon(ep): return max(0.01, min(1.0, 1.0 - math.log10((ep + 1) / 25)))
def get_lr(ep): return max(0.01, min(0.5, 1.0 - math.log10((ep + 1) / 25)))

# 初始化 Q-table
q_table = np.zeros(N_BUCKETS + (2,))
reward_list = []
env = gym.make(ENV_NAME)

# 訓練過程
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    state = get_state(obs)
    total_reward = 0
    done = False
    epsilon = get_epsilon(ep)
    lr = get_lr(ep)

    while not done:
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[state])
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = get_state(next_obs)

        best_q = np.max(q_table[next_state])
        q_table[state + (action,)] += lr * (reward + 0.99 * best_q - q_table[state + (action,)])

        state = next_state
        total_reward += reward

    reward_list.append(total_reward)
    print(f"🎮 Episode {ep + 1} reward: {total_reward:.2f}")

np.save(MODEL_PATH, q_table)

# 畫 reward 圖
plt.plot(reward_list)
plt.title("Q-Table Training Reward (CartPole)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig(PLOT_PATH)

# 錄影（從 Q-table policy）
vec_env = DummyVecEnv([lambda: gym.make(ENV_NAME, render_mode="rgb_array")])
vec_env = VecVideoRecorder(vec_env, VIDEO_PATH, record_video_trigger=lambda x: x == 0,
                           video_length=500, name_prefix="qtable_cartpole_recording")

obs = vec_env.reset()
state = get_state(obs[0])
for _ in range(500):
    action = np.argmax(q_table[state])
    obs, _, dones, _ = vec_env.step([action])
    state = get_state(obs[0])
    if dones[0]:
        break
vec_env.close()
