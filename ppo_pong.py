from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper
import gymnasium as gym
import os
from utils import plot_rewards, record_video

# ✅ 設定參數
ENV_ID = "ALE/Pong-v5"
TOTAL_TIMESTEPS = 100_000
MODEL_PATH = "ppo_pong.zip"
REWARD_PLOT = "ppo_reward_plot.png"
VIDEO_FOLDER = "videos"
VIDEO_LENGTH = 1000

# ✅ 建立 Atari 環境（含預處理）
env = gym.make(ENV_ID, render_mode="rgb_array")
env = AtariWrapper(env)
env = Monitor(env)

# ✅ 建立與訓練 PPO 模型
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./logs/ppo_pong/")
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(MODEL_PATH)

# ✅ reward 曲線
reward_data = env.get_episode_rewards()
plot_rewards(reward_data, save_path=REWARD_PLOT, title="PPO Pong Training Reward")

# ✅ 錄製影片
record_video(MODEL_PATH, ENV_ID, video_length=VIDEO_LENGTH, video_folder=VIDEO_FOLDER, prefix="ppo_pong")

print("✅ PPO 訓練完成、模型儲存、圖表與影片輸出完畢！")
