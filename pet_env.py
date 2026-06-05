import math
import random

class PetSim:
    """Pet Environment Simulation in Python"""
    def __init__(self, canvas_width=600, canvas_height=400):
        self.width = canvas_width
        self.height = canvas_height
        self.diagonal = math.sqrt(self.width ** 2 + self.height ** 2)
        self.step_speed = 0.5 * self.diagonal

        # Room Object Coordinate Locations
        self.food_pos = {"x": 80, "y": 80}
        self.bed_pos = {"x": 520, "y": 80}
        self.wash_pos = {"x": 80, "y": 320}
        self.toy_pos = {"x": 520, "y": 320}
        self.chest_pos = {"x": 300, "y": 340}

        # Action Definitions
        self.ACTIONS = {
            "IDLE": 0,
            "WANDER": 1,
            "EAT": 2,
            "SLEEP_BED": 3,
            "SLEEP_FLOOR": 4,
            "PLAY_TOY": 5,
            "GROOM": 6,
            "WASH": 7,
            "CLEAN_TOY": 8
        }

        # Reset states
        self.reset()

    def reset(self):
        # Mood & Physical States (0.00 to 1.00)
        self.cleanliness = 0.8
        self.fullness = 0.7
        self.stamina = 0.8
        self.boredom = 0.2

        # 2D Coordinates
        self.x = self.width / 2.0
        self.y = self.height / 2.0

        # Toy & Command statuses
        self.toy_placed = True
        self.command_pending = False
        self.carrying_toy = False

        # Wander target (if any)
        self.wander_target = None

        # Personality Parameters (Default 0.50)
        self.personality = {
            "activeness": 0.5,
            "gluttony": 0.5,
            "patience": 0.5,
            "curiosity": 0.5,
            "loyalty": 0.5
        }

        # UI & Logging parameters
        self.last_action = 0
        self.last_reward = 0.0
        self.last_action_status = "초기화됨"

    def set_personality(self, activeness, gluttony, patience, curiosity, loyalty):
        self.personality["activeness"] = activeness
        self.personality["gluttony"] = gluttony
        self.personality["patience"] = patience
        self.personality["curiosity"] = curiosity
        self.personality["loyalty"] = loyalty

    def get_state_vector(self):
        """Get 14-dimensional normalized state vector"""
        d_food = self.get_normalized_dist(self.food_pos)
        d_bed = self.get_normalized_dist(self.bed_pos)
        d_toy = self.get_normalized_dist(self.chest_pos if self.carrying_toy else self.toy_pos) if self.toy_placed else 1.0
        d_wash = self.get_normalized_dist(self.wash_pos)

        return [
            self.cleanliness,
            self.fullness,
            self.stamina,
            self.boredom,
            d_food,
            d_bed,
            d_toy,
            d_wash,
            1.0 if self.command_pending else 0.0,
            self.personality["activeness"],
            self.personality["gluttony"],
            self.personality["patience"],
            self.personality["curiosity"],
            self.personality["loyalty"]
        ]

    def get_normalized_dist(self, target):
        dx = self.x - target["x"]
        dy = self.y - target["y"]
        dist = math.sqrt(dx ** 2 + dy ** 2)
        return min(1.0, dist / self.diagonal)

    def move_towards(self, target_x, target_y, speed=15):
        """Move pet step-by-step and return True if arrived"""
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist <= speed:
            self.x = float(target_x)
            self.y = float(target_y)
            return True

        self.x += (dx / dist) * speed
        self.y += (dy / dist) * speed
        return False

    def step(self, action):
        """Simulate single timestep step transition and calculate rewards"""
        self.last_action = action
        reward = 0.0
        status = ""

        # Ambient decay of states on each step
        self.cleanliness = max(0.0, self.cleanliness - 0.005)
        self.fullness = max(0.0, self.fullness - 0.01)
        self.stamina = max(0.0, self.stamina - 0.003)
        self.boredom = min(1.0, self.boredom + 0.007)

        # Core parameters for reward evaluation
        S = self.stamina
        F = self.fullness
        C = self.cleanliness
        B = self.boredom

        P_active = self.personality["activeness"]
        P_eat = self.personality["gluttony"]
        P_patience = self.personality["patience"]
        P_curious = self.personality["curiosity"]
        P_loyal = self.personality["loyalty"]

        d_food = self.get_normalized_dist(self.food_pos)
        d_bed = self.get_normalized_dist(self.bed_pos)
        d_toy = self.get_normalized_dist(self.toy_pos) if self.toy_placed else 1.0
        d_wash = self.get_normalized_dist(self.wash_pos)
        d_chest = self.get_normalized_dist(self.chest_pos)

        is_close_threshold = 0.06 # Threshold for arrival (~36px)

        # Idle
        if action == self.ACTIONS["IDLE"]:
            status = "가만히 쉬는 중"
            self.boredom = min(1.0, self.boredom + 0.05)
            self.stamina = min(1.0, self.stamina + 0.01)
            reward = 0.05 + 0.2 * P_patience - 0.1 * P_active + 0.15 * (1.0 - S) - 0.2 * B
            self.carrying_toy = False

        # Wander
        elif action == self.ACTIONS["WANDER"]:
            status = "방안 배회 중"
            if self.wander_target is None:
                self.wander_target = {
                    "x": random.uniform(50, self.width - 50),
                    "y": random.uniform(50, self.height - 50)
                }
            arrived = self.move_towards(self.wander_target["x"], self.wander_target["y"], speed=self.step_speed)
            if arrived:
                self.wander_target = None

            self.stamina = max(0.0, self.stamina - 0.02)
            self.fullness = max(0.0, self.fullness - 0.01)
            self.boredom = max(0.0, self.boredom - 0.08)
            reward = 0.05 + 0.3 * P_active + 0.15 * P_curious + 0.2 * B - 0.2 * (1.0 - S)
            self.carrying_toy = False

        # Eat
        elif action == self.ACTIONS["EAT"]:
            if d_food > is_close_threshold:
                status = "밥그릇으로 이동 중"
                self.move_towards(self.food_pos["x"], self.food_pos["y"], speed=self.step_speed)
                reward = 0.1 * (1.0 - d_food) - 0.02
            else:
                status = "얌얌 밥 먹는 중"
                self.fullness = min(1.0, self.fullness + 0.30)
                self.cleanliness = max(0.0, self.cleanliness - 0.05)
                self.stamina = max(0.0, self.stamina - 0.01)
                reward = 0.5 * (1.0 - F) + 0.5 * P_eat
                if F > 0.9:
                    reward -= 0.4
            self.carrying_toy = False

        # Sleep Bed
        elif action == self.ACTIONS["SLEEP_BED"]:
            if d_bed > is_close_threshold:
                status = "침대로 이동 중"
                self.move_towards(self.bed_pos["x"], self.bed_pos["y"], speed=self.step_speed)
                reward = 0.1 * (1.0 - d_bed) - 0.02
            else:
                status = "침대에서 꿀잠 자는 중"
                self.stamina = min(1.0, self.stamina + 0.25)
                reward = 0.6 * (1.0 - S) + 0.15 * P_patience
                if S > 0.9:
                    reward -= 0.4
            self.carrying_toy = False

        # Sleep Floor
        elif action == self.ACTIONS["SLEEP_FLOOR"]:
            status = "바닥에서 엎드려 자는 중"
            self.stamina = min(1.0, self.stamina + 0.12)
            self.cleanliness = max(0.0, self.cleanliness - 0.08)
            reward = 0.35 * (1.0 - S) - 0.1 * P_patience - 0.15 * C
            self.carrying_toy = False

        # Play Toy
        elif action == self.ACTIONS["PLAY_TOY"]:
            if not self.toy_placed:
                status = "놀 장난감이 없어서 보채는 중"
                reward = -0.5
            elif d_toy > is_close_threshold:
                status = "장난감으로 이동 중"
                self.move_towards(self.toy_pos["x"], self.toy_pos["y"], speed=self.step_speed)
                reward = 0.1 * (1.0 - d_toy) - 0.02
            else:
                status = "장난감 신나게 가지고 노는 중"
                self.boredom = max(0.0, self.boredom - 0.25)
                self.stamina = max(0.0, self.stamina - 0.05)
                self.fullness = max(0.0, self.fullness - 0.02)
                self.cleanliness = max(0.0, self.cleanliness - 0.05)
                
                # Base reward based on boredom and personality
                reward = 0.6 * B + 0.3 * P_curious + 0.1 * P_active
                if B < 0.2:
                    reward -= 0.4  # Penalty for playing when already fully entertained
            self.carrying_toy = False

        # Groom
        elif action == self.ACTIONS["GROOM"]:
            status = "손발 그루밍(햝기) 중"
            self.cleanliness = min(1.0, self.cleanliness + 0.10)
            self.boredom = max(0.0, self.boredom - 0.05)
            self.stamina = max(0.0, self.stamina - 0.01)
            reward = 0.25 * (1.0 - C) + 0.08 * B - 0.1 * P_active
            self.carrying_toy = False

        # Wash
        elif action == self.ACTIONS["WASH"]:
            if d_wash > is_close_threshold:
                status = "욕조로 이동 중"
                self.move_towards(self.wash_pos["x"], self.wash_pos["y"], speed=self.step_speed)
                reward = 0.1 * (1.0 - d_wash) - 0.02
            else:
                status = "욕조에서 물장구 치며 씻는 중"
                self.cleanliness = min(1.0, self.cleanliness + 0.35)
                self.stamina = max(0.0, self.stamina - 0.04)
                self.fullness = max(0.0, self.fullness - 0.02)
                reward = 0.6 * (1.0 - C) + 0.15 * P_patience
                if C > 0.9:
                    reward -= 0.3
            self.carrying_toy = False

        # Clean Toy
        elif action == self.ACTIONS["CLEAN_TOY"]:
            if not self.toy_placed:
                status = "정리할 장난감이 없음"
                reward = -0.5
            elif not self.carrying_toy:
                if d_toy > is_close_threshold:
                    status = "장난감 집으러 이동 중"
                    self.move_towards(self.toy_pos["x"], self.toy_pos["y"], speed=self.step_speed)
                    reward = 0.1 * (1.0 - d_toy)
                else:
                    status = "장난감 집어 들음"
                    self.carrying_toy = True
                    reward = 0.1
            else:
                if d_chest > is_close_threshold:
                    status = "장난감 상자로 운반 중"
                    self.move_towards(self.chest_pos["x"], self.chest_pos["y"], speed=self.step_speed)
                    reward = 0.1 * (1.0 - d_chest)
                else:
                    status = "장난감 정리 완료!"
                    self.toy_placed = False
                    self.carrying_toy = False
                    self.command_pending = False
                    
                    self.stamina = max(0.0, self.stamina - 0.05)
                    self.fullness = max(0.0, self.fullness - 0.02)
                    self.boredom = min(1.0, self.boredom + 0.05)
                    
                    if self.command_pending:
                        reward = 0.8 * P_loyal if P_loyal >= 0.5 else -0.3 * (1.0 - P_loyal)
                    else:
                        reward = -0.1

        # Apply loyalty penalties/rewards if owner issued a command
        if self.command_pending:
            if action == self.ACTIONS["CLEAN_TOY"]:
                if P_loyal >= 0.5:
                    reward += 0.4 * P_loyal
                else:
                    reward -= 0.2 * (1.0 - P_loyal)
            else:
                if P_loyal >= 0.5:
                    reward -= 0.5 * P_loyal
                else:
                    reward += 0.3 * (1.0 - P_loyal)

        # Keep state values bounded
        self.cleanliness = min(1.0, max(0.0, self.cleanliness))
        self.fullness = min(1.0, max(0.0, self.fullness))
        self.stamina = min(1.0, max(0.0, self.stamina))
        self.boredom = min(1.0, max(0.0, self.boredom))

        self.last_reward = float(reward)
        self.last_action_status = status

        return reward

    def place_toy(self):
        """Spawn a new toy inside room boundaries"""
        if not self.toy_placed:
            self.toy_placed = True
            self.carrying_toy = False
            self.toy_pos = {
                "x": random.uniform(100, self.width - 100),
                "y": random.uniform(100, self.height - 100)
            }
            return True
        return False
