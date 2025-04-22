import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

ENV_NAME = "CartPole-v1"
VIDEO_PATH = "videos_handmade"
PLOT_PATH = "handmade_reward_plot.png"
N_EPISODES = 300

reward_list = []

# 手寫策略：只根據 pole 的角度進行左右移動
def choose_action(obs):
    _, _, angle, _ = obs
    return 0 if angle < 0 else 1

# 環境初始化
env = gym.make(ENV_NAME)

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = choose_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    reward_list.append(total_reward)
    print(f"🎮 Episode {ep + 1} reward: {total_reward:.2f}")

env.close()

# 畫 reward 圖
plt.plot(reward_list)
plt.title("Hand-Made Policy Reward (CartPole)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig(PLOT_PATH)
print(f"📊 reward 曲線圖已儲存為 {PLOT_PATH}")

# 錄影（手寫策略）
vec_env = DummyVecEnv([lambda: gym.make(ENV_NAME, render_mode="rgb_array")])
vec_env = VecVideoRecorder(vec_env, VIDEO_PATH, record_video_trigger=lambda x: x == 0,
                           video_length=500, name_prefix="handmade_cartpole_recording")
obs = vec_env.reset()

for _ in range(500):
    action = choose_action(obs[0])
    obs, _, dones, _ = vec_env.step([action])
    if dones[0]:
        break
vec_env.close()
print(f"🎥 影片已儲存至 {VIDEO_PATH}")
