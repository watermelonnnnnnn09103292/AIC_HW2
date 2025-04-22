import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

ENV_NAME = "CartPole-v1"
VIDEO_PATH = "videos_random"
PLOT_PATH = "random_reward_plot.png"
REWARD_LIST = []
N_EPISODES = 300

# 訓練過程（無錄影）
env = gym.make(ENV_NAME, render_mode=None)
for i_episode in range(N_EPISODES):
    obs, _ = env.reset()
    total_reward = 0
    for t in range(500):
        action = env.action_space.sample()  # 隨機選擇動作
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if done:
            break
    print(f"🎮 Episode {i_episode+1} reward: {total_reward:.2f}")
    REWARD_LIST.append(total_reward)
env.close()

# 畫 reward 曲線圖
plt.plot(REWARD_LIST, color="gray")
plt.title("Random Action Reward (CartPole)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig(PLOT_PATH)
plt.close()
print(f"📊 reward 曲線圖已儲存為 {PLOT_PATH}")

# 錄影（Random Action 策略）
vec_env = DummyVecEnv([lambda: gym.make(ENV_NAME, render_mode="rgb_array")])
vec_env = VecVideoRecorder(
    vec_env,
    VIDEO_PATH,
    record_video_trigger=lambda x: x == 0,  # 第 0 回合錄影
    video_length=500,
    name_prefix="random_cartpole_recording"
)
obs = vec_env.reset()
for _ in range(500):
    action = vec_env.action_space.sample()
    obs, _, dones, _ = vec_env.step([action])
    if dones[0]:
        break
vec_env.close()
print(f"🎥 影片已儲存至 {VIDEO_PATH}")
