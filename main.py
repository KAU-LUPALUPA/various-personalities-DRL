import pygame
import sys
import numpy as np

# Import custom modules
from pet_env import PetSim
from dqn_agent import DQNAgent
from simulator import PygameSimulator

def main():
    # Initialize Pygame
    pygame.init()
    
    # Environment and Agent sizes
    canvas_w = 600
    canvas_h = 400
    pet = PetSim(canvas_w, canvas_h)
    
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

    # Initialize Simulator
    sim = PygameSimulator(pet, agent)

    # Load previously saved model weights if they exist
    if agent.load_model():
        sim.status_log = "이전 학습된 펫의 모델을 성공적으로 불러왔습니다."
        sim.status_log_color = (0, 235, 199)

    # Synchronize Sliders with Pet initial personalities
    def update_pet_personalities():
        pet.set_personality(
            activeness=sim.sliders["activeness"].current_val,
            gluttony=sim.sliders["gluttony"].current_val,
            patience=sim.sliders["patience"].current_val,
            curiosity=sim.sliders["curiosity"].current_val,
            loyalty=sim.sliders["loyalty"].current_val
        )
    update_pet_personalities()

    # Time tracking variables
    clock = pygame.time.Clock()
    step_count = 0
    episode_count = 0
    episode_steps = 0
    max_steps_per_episode = 300
    current_episode_reward = 0.0
    reward_history = []
    
    is_running = True
    sim_speed = 1
    pretrain_triggered = False
    last_step_time = pygame.time.get_ticks()

    # Main Game Loop
    while True:
        # 1. Handle UI Events
        events = sim.handle_events()
        for event in events:
            if isinstance(event, tuple):
                if event[0] == "personality_change":
                    update_pet_personalities()
                elif event[0] == "speed_change":
                    sim_speed = int(event[1])
            else:
                if event == "cmd_clean":
                    pet.command_pending = True
                    sim.status_log = "정리 명령이 접수되었습니다! 충성도에 따라 반응합니다."
                    sim.status_log_color = (162, 155, 254)
                elif event == "cmd_place_toy":
                    if pet.place_toy():
                        sim.status_log = "바닥에 새로운 장난감을 놔주었습니다. 펫이 놀 수 있습니다."
                        sim.status_log_color = (164, 176, 190)
                        sim.add_particles(pet.toy_pos["x"], pet.toy_pos["y"], "clean", 12)
                elif event == "cmd_pretrain":
                    pretrain_triggered = True
                elif event == "cmd_toggle_pause":
                    is_running = not is_running
                    last_step_time = pygame.time.get_ticks()
                elif event == "cmd_reset":
                    pet.reset()
                    agent.epsilon = 0.4
                    step_count = 0
                    episode_count = 0
                    episode_steps = 0
                    current_episode_reward = 0.0
                    reward_history.clear()
                    sim.particles.clear()
                    sim.draw_x = pet.x
                    sim.draw_y = pet.y
                    update_pet_personalities()
                    sim.status_log = "초기화되었습니다. 슬라이더로 성격을 직접 조절해 보세요."
                    sim.status_log_color = (164, 176, 190)
                    last_step_time = pygame.time.get_ticks()
                elif event == "cmd_quit":
                    agent.save_model()
                    pygame.quit()
                    sys.exit()

        # 2. Pre-training Routine
        if pretrain_triggered:
            # Draw temporary loading text
            sim.screen.fill((11, 13, 25))
            loading_txt = sim.font_section.render("DQN 펫 두뇌 고속 고밀도 학습 중... (10000 걸음)", True, (0, 235, 199))
            sim.screen.blit(loading_txt, (sim.screen_width // 2 - loading_txt.get_width() // 2, sim.screen_height // 2 - 20))
            pygame.display.flip()

            # Save user's current slider values
            saved_active = sim.sliders["activeness"].current_val
            saved_gluttony = sim.sliders["gluttony"].current_val
            saved_patience = sim.sliders["patience"].current_val
            saved_curiosity = sim.sliders["curiosity"].current_val
            saved_loyalty = sim.sliders["loyalty"].current_val

            pretrain_steps = 10000
            for pretrain_i in range(pretrain_steps):
                # Randomize personality every 150 steps to cover the full parameter space
                if pretrain_i % 150 == 0:
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

                step_count += 1
                episode_steps += 1
                current_episode_reward += r

                if episode_steps >= max_steps_per_episode:
                    reward_history.append(current_episode_reward / max_steps_per_episode)
                    if len(reward_history) > 50:
                        reward_history.pop(0)
                    current_episode_reward = 0.0
                    episode_steps = 0
                    episode_count += 1

            # Restore user's slider settings after pre-training completes
            pet.set_personality(saved_active, saved_gluttony, saved_patience, saved_curiosity, saved_loyalty)

            # Decay epsilon
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.3)
            pretrain_triggered = False
            sim.status_log = f"사전 학습 완료! ({pretrain_steps}걸음 학습됨. Epsilon: {agent.epsilon:.2f})"
            sim.status_log_color = (0, 235, 199)
            
            # Save pretrained weights
            agent.save_model()
            
            last_step_time = pygame.time.get_ticks()
            sim.draw_x = pet.x
            sim.draw_y = pet.y

        # 3. Regular Simulation Step
        elif is_running:
            current_time = pygame.time.get_ticks()
            step_interval = 1000.0 / sim_speed
            
            if current_time - last_step_time >= step_interval:
                steps_to_run = int((current_time - last_step_time) // step_interval)
                steps_to_run = min(steps_to_run, 5)  # Cap catch-up steps
                
                for _ in range(steps_to_run):
                    s = pet.get_state_vector()
                    a = agent.act(s)
                    r = pet.step(a)
                    next_s = pet.get_state_vector()
                    
                    agent.remember(s, a, r, next_s, False)
                    agent.train()

                    step_count += 1
                    episode_steps += 1
                    current_episode_reward += r

                    # Spawning action particles
                    if a == pet.ACTIONS["EAT"] and pet.get_normalized_dist(pet.food_pos) < 0.07:
                        sim.add_particles(pet.x, pet.y, "eat", 3)
                    elif (a == pet.ACTIONS["SLEEP_BED"] and pet.get_normalized_dist(pet.bed_pos) < 0.07) or a == pet.ACTIONS["SLEEP_FLOOR"]:
                        sim.add_particles(pet.x, pet.y - 12, "sleep", 2)
                    elif a == pet.ACTIONS["WASH"] and pet.get_normalized_dist(pet.wash_pos) < 0.07:
                        sim.add_particles(pet.x, pet.y, "wash", 4)
                    elif a == pet.ACTIONS["PLAY_TOY"] and pet.toy_placed and pet.get_normalized_dist(pet.toy_pos) < 0.07:
                        sim.add_particles(pet.x, pet.y, "clean", 3)

                    # Episode boundary check
                    if episode_steps >= max_steps_per_episode:
                        reward_history.append(current_episode_reward / max_steps_per_episode)
                        if len(reward_history) > 50:
                            reward_history.pop(0)
                        
                        current_episode_reward = 0.0
                        episode_steps = 0
                        episode_count += 1
                        
                        # Auto-save model weights every 5 episodes
                        if episode_count % 5 == 0:
                            agent.save_model()
                        
                last_step_time = current_time
        
        else:
            # Keep updating last_step_time while paused to avoid catch-up jump
            last_step_time = pygame.time.get_ticks()

        # 4. Render and Flip
        sim.update_particles()
        sim.draw(step_count, episode_count, is_running)
        sim.draw_reward_history(reward_history)
        pygame.display.flip()
        
        # Lock FPS to 60
        clock.tick(60)

if __name__ == "__main__":
    main()
