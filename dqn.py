import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

CHEAT = True
MODEL_PATH = "dqn_cartpole_model.pth"
PLOT_PATH = "cartpole_reward_plot.png"
VIDEO_PATH = "videos"
REWARD_LIST = []

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.out(x)

class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden), Net(n_states, n_actions, n_hidden)
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0

        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity

    def choose_action(self, state):
        x = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0)
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            actions_value = self.eval_net(x)
            return torch.argmax(actions_value, dim=1).item()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    dqn = DQN(n_states, n_actions, n_hidden=50, batch_size=32, lr=0.01,
              epsilon=0.1, gamma=0.9, target_replace_iter=100, memory_capacity=2000)

    n_episodes = 300 if CHEAT else 1000

    for i_episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        t = 0
        while True:
            action = dqn.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if CHEAT:
                x, v, theta, omega = next_state
                r1 = (env.unwrapped.x_threshold - abs(x)) / env.unwrapped.x_threshold - 0.8
                r2 = (env.unwrapped.theta_threshold_radians - abs(theta)) / env.unwrapped.theta_threshold_radians - 0.5
                reward = r1 + r2

            dqn.store_transition(state, action, reward, next_state)
            total_reward += reward

            if dqn.memory_counter > dqn.memory_capacity:
                dqn.learn()

            state = next_state
            t += 1
            if done:
                print(f"🎮 Episode {i_episode+1} finished after {t} timesteps, total rewards: {total_reward:.2f}")
                REWARD_LIST.append(total_reward)
                break

    # 儲存 model
    torch.save(dqn.eval_net.state_dict(), MODEL_PATH)
    print(f"✅ 模型已儲存為 {MODEL_PATH}")

    # 畫 reward 曲線
    plt.plot(REWARD_LIST)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Reward (CartPole)")
    plt.savefig(PLOT_PATH)
    print(f"📊 reward 曲線圖已儲存為 {PLOT_PATH}")

    # 錄製影片
    vec_env = DummyVecEnv([lambda: gym.make("CartPole-v1", render_mode="rgb_array")])
    vec_env = VecVideoRecorder(vec_env, VIDEO_PATH, record_video_trigger=lambda x: x == 0,
                               video_length=500, name_prefix="dqn_cartpole_recording")
    obs = vec_env.reset()
    for _ in range(500):
        action = dqn.choose_action(obs[0])
        obs, _, dones, _ = vec_env.step([action])
        if dones[0]:
            break
    vec_env.close()
    print(f"🎥 影片已儲存至 {VIDEO_PATH}")
