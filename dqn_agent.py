import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    """Deep Q-Network Model (Multi-Layer Perceptron)"""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
        # Initialize weights using Xavier (Glorot) initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience Replay Buffer to break correlation between consecutive states"""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size=14, action_size=9, lr=0.003, gamma=0.90, 
                 epsilon=0.4, epsilon_decay=0.9995, epsilon_min=0.05, 
                 batch_size=64, target_update_interval=150):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        
        # Device configuration (Force GPU if available, with clear console indicators)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("\n" + "="*50)
            print("★ [GPU (CUDA) 가속 학습 모드]가 정상 활성화되었습니다. ★")
            print(f"사용 GPU 디바이스: {torch.cuda.get_device_name(0)}")
            print("="*50 + "\n")
        else:
            self.device = torch.device("cpu")
            print("\n" + "="*50)
            print("⚠ [경고] GPU (CUDA)를 사용할 수 없어 CPU 모드로 실행합니다.")
            print("NVIDIA 그래픽 카드 드라이버 및 PyTorch CUDA 구성을 확인하세요.")
            print("="*50 + "\n")
        
        # Networks
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target net to evaluation mode
        
        # Optimizer and Loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss for training stability
        
        # Replay Buffer
        self.memory = ReplayBuffer(max_size=10000)
        self.steps_count = 0
        self.recent_losses = []

    def act(self, state, force_exploit=False):
        """Epsilon-Greedy action selection"""
        if not force_exploit and random.random() < self.epsilon:
            # Explore
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def get_q_values(self, state):
        """Get raw Q-values for visual bar graphs in Pygame"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.cpu().numpy()[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        """Train on a batch sampled from Replay Memory"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch Tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Get Q-values for current states
        current_q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Compute target Q-values using Target Network (Double DQN can be added, but standard DQN is fine here)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states_t).max(dim=1)[0]
            target_q_values = rewards_t + (self.gamma * max_next_q_values * (1 - dones_t))

        # Loss calculation (Huber Loss)
        loss = self.loss_fn(current_q_values, target_q_values)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps_count += 1
        
        # Target network update
        if self.steps_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.recent_losses.append(loss_val)
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)

        return loss_val

    def save_model(self, filepath='pet_dqn_model.pth'):
        """Save neural network weights to disk"""
        try:
            torch.save(self.policy_net.state_dict(), filepath)
            print(f"★ 모델 가중치가 '{filepath}'에 자동 저장되었습니다. ★")
            return True
        except Exception as e:
            print(f"모델 저장 실패: {e}")
            return False

    def load_model(self, filepath='pet_dqn_model.pth'):
        """Load neural network weights from disk if file exists"""
        if os.path.exists(filepath):
            try:
                self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                # Decay epsilon since the network starts with learned weights
                self.epsilon = self.epsilon_min
                print(f"★ 기존에 학습된 모델 '{filepath}'를 불러왔습니다. (Epsilon: {self.epsilon:.2f}) ★")
                return True
            except Exception as e:
                print(f"모델 불러오기 오류: {e}")
        return False
