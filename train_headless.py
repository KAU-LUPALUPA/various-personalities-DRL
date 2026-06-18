import numpy as np
import torch
from pet_env import PetSim
from dqn_agent import DQNAgent

def main():
    print("★ [헤드리스 고속 학습 시작] 새로운 보상 규칙으로 DQN 펫 두뇌를 학습시킵니다... ★")
    
    # Initialize env and agent
    pet = PetSim()
    agent = DQNAgent(
        state_size=14,
        action_size=9,
        lr=0.003,
        gamma=0.90,
        epsilon=0.4,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
        batch_size=64,
        target_update_interval=150
    )
    
    # Try to load existing model first to fine-tune it (optional, but training from scratch ensures complete adaptation)
    # Let's train from scratch to ensure the agent completely adapts to the corrected reward signals.
    print("성격 분포 학습을 위해 가중치를 초기화하고 새로 학습을 진행합니다.")

    total_steps = 25000
    episode_steps = 0
    max_steps_per_episode = 300
    current_reward = 0.0
    
    for step_i in range(total_steps):
        # Randomize personality every 150 steps to cover the full parameter space
        if step_i % 150 == 0:
            pet.set_personality(
                activeness=np.random.rand(),
                gluttony=np.random.rand(),
                patience=np.random.rand(),
                curiosity=np.random.rand(),
                loyalty=np.random.rand()
            )

        s = pet.get_state_vector()
        
        # Place toy with small probability to avoid environment lock
        if not pet.toy_placed and np.random.rand() < 0.05:
            pet.place_toy()
        if not pet.command_pending and pet.toy_placed and np.random.rand() < 0.03:
            pet.command_pending = True

        a = agent.act(s)
        r = pet.step(a)
        next_s = pet.get_state_vector()
        
        agent.remember(s, a, r, next_s, False)
        agent.train()

        current_reward += r
        episode_steps += 1

        if episode_steps >= max_steps_per_episode:
            episode_steps = 0
            current_reward = 0.0

        if (step_i + 1) % 5000 == 0:
            print(f"학습 진행률: {step_i + 1}/{total_steps} 걸음 완료 (Epsilon: {agent.epsilon:.3f})")

    # Decay epsilon further for exploit phase evaluation
    agent.epsilon = agent.epsilon_min
    
    # Save the updated model
    agent.save_model('pet_dqn_model.pth')
    print("★ 학습 완료 및 'pet_dqn_model.pth' 저장 성공! ★")

if __name__ == "__main__":
    main()
