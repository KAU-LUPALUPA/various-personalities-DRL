import pygame
import random
import math
import numpy as np

# Color Palette (Dark Mode & Premium Accents)
BG_DARK = (11, 13, 25)
PANEL_BG = (20, 24, 46)
PANEL_BORDER = (40, 45, 75)
TEXT_PRIMARY = (241, 242, 246)
TEXT_SECONDARY = (164, 176, 190)

ACCENT = (108, 92, 231)
COLOR_CLEAN = (0, 235, 199)
COLOR_FULL = (255, 84, 112)
COLOR_STAMINA = (255, 209, 102)
COLOR_BORED = (162, 155, 254)

SUCCESS = (46, 213, 115)
DANGER = (255, 71, 87)
SHADOW = (5, 5, 12)

class Button:
    """Custom GUI Button for Pygame"""
    def __init__(self, x, y, width, height, text, bg_color=ACCENT, text_color=TEXT_PRIMARY):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.bg_color = bg_color
        self.text_color = text_color
        self.is_hovered = False
        self.is_disabled = False

    def draw(self, screen, font):
        color = self.bg_color
        if self.is_disabled:
            color = (50, 52, 70)
        elif self.is_hovered:
            # Lighten the color slightly on hover
            color = tuple(min(255, c + 30) for c in self.bg_color)

        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, PANEL_BORDER, self.rect, width=1, border_radius=8)

        text_surf = font.render(self.text, True, self.text_color if not self.is_disabled else TEXT_SECONDARY)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if self.is_disabled:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                return True
        return False

class Slider:
    """Custom GUI Slider for Pygame"""
    def __init__(self, x, y, width, height, label, min_val=0.0, max_val=1.0, initial_val=0.5):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = initial_val
        self.handle_radius = 8
        self.is_dragging = False
        
        # Calculate handle position
        self.handle_x = self.rect.x + int((self.current_val - self.min_val) / (self.max_val - self.min_val) * self.rect.width)

    def draw(self, screen, font_lbl, font_val):
        # Draw Label
        lbl_surf = font_lbl.render(self.label, True, TEXT_SECONDARY)
        screen.blit(lbl_surf, (self.rect.x, self.rect.y - 20))

        # Draw Value
        val_surf = font_val.render(f"{self.current_val:.2f}", True, TEXT_PRIMARY)
        screen.blit(val_surf, (self.rect.right - val_surf.get_width(), self.rect.y - 20))

        # Draw Track
        pygame.draw.rect(screen, (35, 39, 66), self.rect, border_radius=4)
        
        # Draw Filled Track
        filled_rect = pygame.Rect(self.rect.x, self.rect.y, self.handle_x - self.rect.x, self.rect.height)
        pygame.draw.rect(screen, ACCENT, filled_rect, border_radius=4)

        # Draw Handle
        handle_color = (130, 115, 250) if self.is_dragging else ACCENT
        pygame.draw.circle(screen, handle_color, (self.handle_x, self.rect.centery), self.handle_radius)
        pygame.draw.circle(screen, TEXT_PRIMARY, (self.handle_x, self.rect.centery), self.handle_radius - 4)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = event.pos
                # Check collision with handle or track
                dist = math.sqrt((mouse_pos[0] - self.handle_x) ** 2 + (mouse_pos[1] - self.rect.centery) ** 2)
                if dist <= self.handle_radius or self.rect.collidepoint(mouse_pos):
                    self.is_dragging = True
                    self.update_value(mouse_pos[0])
                    return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.is_dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.is_dragging:
                self.update_value(event.pos[0])
                return True
        return False

    def update_value(self, mouse_x):
        mouse_x = max(self.rect.x, min(self.rect.right, mouse_x))
        self.handle_x = mouse_x
        pct = (mouse_x - self.rect.x) / self.rect.width
        self.current_val = self.min_val + pct * (self.max_val - self.min_val)

class PygameSimulator:
    """Simulator Drawing and UI Management Module"""
    def __init__(self, pet_sim, agent):
        self.pet = pet_sim
        self.agent = agent

        # Setup Pygame screen size (1000 x 600)
        self.screen_width = 1000
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("DRL 펫 성격 시뮬레이터 (PyTorch & Pygame)")

        # Load fonts supporting Korean
        font_choices = ["malgungothic", "nanumgothic", "gulim", "segoeui", "arial"]
        self.font_title = pygame.font.SysFont(font_choices, 24, bold=True)
        self.font_section = pygame.font.SysFont(font_choices, 18, bold=True)
        self.font_body = pygame.font.SysFont(font_choices, 14)
        self.font_body_bold = pygame.font.SysFont(font_choices, 14, bold=True)
        self.font_small = pygame.font.SysFont(font_choices, 11)

        # UI Control Widgets Setup
        # Buttons (Row 1)
        self.btn_clean = Button(20, 420, 175, 35, "장난감 정리 시키기", bg_color=ACCENT)
        self.btn_toy = Button(210, 420, 175, 35, "장난감 주기", bg_color=PANEL_BG)
        self.btn_pretrain = Button(405, 420, 175, 35, "뇌 사전 학습 (+10000)", bg_color=COLOR_CLEAN, text_color=BG_DARK)
        
        # Buttons (Row 2)
        self.btn_pause = Button(20, 470, 175, 35, "시뮬레이션 일시정지", bg_color=ACCENT)
        self.btn_reset = Button(210, 470, 175, 35, "초기화", bg_color=PANEL_BG)
        
        # Speed slider in Buttons panel
        self.speed_slider = Slider(405, 485, 175, 8, "시뮬레이션 속도", min_val=1.0, max_val=20.0, initial_val=1.0)
        self.speed_slider.handle_radius = 6

        # Personality Sliders Y: 70 to 220: Spacing 30
        self.sliders = {
            "activeness": Slider(680, 80, 200, 6, "활발함 (Activeness)", initial_val=0.50),
            "gluttony": Slider(680, 125, 200, 6, "먹성 (Gluttony)", initial_val=0.50),
            "patience": Slider(680, 170, 200, 6, "인내심 (Patience)", initial_val=0.50),
            "curiosity": Slider(680, 215, 200, 6, "호기심 (Curiosity)", initial_val=0.50),
            "loyalty": Slider(680, 260, 200, 6, "주인 충성도 (Loyalty)", initial_val=0.50)
        }

        # Status text box description log
        self.status_log = "바닥에 새로운 장난감을 놔주었습니다. 펫이 놀 수 있습니다."
        self.status_log_color = TEXT_SECONDARY

        # Particle animations
        self.particles = []
        self.draw_x = self.pet.x
        self.draw_y = self.pet.y
        self.action_names = [
            "대기 (Idle)",
            "배회 (Wander)",
            "식사 (Eat)",
            "침대 수면 (Sleep)",
            "바닥 수면 (Floor)",
            "놀이 (Play)",
            "그루밍 (Groom)",
            "목욕 (Wash)",
            "정리 (Clean)"
        ]

    def add_particles(self, x, y, particle_type, count=5):
        if particle_type == 'eat':
            colors = [(255, 209, 102), (255, 84, 112), (255, 255, 255)]
        elif particle_type == 'sleep':
            colors = [(255, 255, 255), (162, 155, 254), (129, 236, 236)]
        elif particle_type == 'wash':
            colors = [(129, 236, 236), (116, 185, 255), (9, 132, 227)]
        elif particle_type == 'clean':
            colors = [(0, 235, 199), (0, 184, 148), (255, 234, 167)]
        else:
            colors = [(255, 118, 117), (253, 203, 110), (255, 234, 167)]

        for _ in range(count):
            self.particles.append({
                "x": x,
                "y": y,
                "vx": random.uniform(-2, 2),
                "vy": random.uniform(-3, -0.5),
                "radius": random.uniform(2, 5),
                "color": random.choice(colors),
                "alpha": 1.0,
                "decay": random.uniform(0.015, 0.035),
                "type": particle_type
            })

    def update_particles(self):
        active_particles = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["alpha"] -= p["decay"]
            if p["alpha"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def draw(self, step_count, episode_count, is_running):
        # Smooth gliding position interpolation (visual only)
        dx = self.pet.x - self.draw_x
        dy = self.pet.y - self.draw_y
        self.draw_x += dx * 0.08
        self.draw_y += dy * 0.08

        # 1. Clear Screen
        self.screen.fill(BG_DARK)

        # 2. Draw Left Side Panel Borders & Background
        # Left simulator background Y: 0 to 400
        sim_rect = pygame.Rect(0, 0, 600, 400)
        pygame.draw.rect(self.screen, (20, 23, 39), sim_rect)
        
        # Grid lines in room
        grid_size = 40
        for x in range(0, 600, grid_size):
            pygame.draw.line(self.screen, (25, 28, 48), (x, 0), (x, 400), 1)
        for y in range(0, 400, grid_size):
            pygame.draw.line(self.screen, (25, 28, 48), (0, y), (600, y), 1)

        # Draw Object bases & details in room
        def draw_object_base(pos, radius, label):
            pygame.draw.circle(self.screen, (255, 255, 255, 10), (pos["x"], pos["y"]), radius)
            pygame.draw.circle(self.screen, (60, 65, 95), (pos["x"], pos["y"]), radius, width=1)
            lbl = self.font_small.render(label, True, TEXT_SECONDARY)
            self.screen.blit(lbl, (pos["x"] - lbl.get_width() // 2, pos["y"] + radius + 4))

        # Bases
        draw_object_base(self.pet.food_pos, 28, "밥그릇 (Food)")
        draw_object_base(self.pet.bed_pos, 32, "침대 (Bed)")
        draw_object_base(self.pet.wash_pos, 30, "욕조 (Bath)")
        draw_object_base(self.pet.chest_pos, 28, "장난감 상자")

        # Bowl Content
        pygame.draw.circle(self.screen, (255, 118, 117), (self.pet.food_pos["x"], self.pet.food_pos["y"]), 14)
        pygame.draw.circle(self.screen, (255, 234, 167), (self.pet.food_pos["x"] - 2, self.pet.food_pos["y"] - 2), 7)

        # Bed Details
        pygame.draw.rect(self.screen, (116, 185, 255), (self.pet.bed_pos["x"] - 22, self.pet.bed_pos["y"] - 14, 44, 28))
        pygame.draw.rect(self.screen, (255, 255, 255), (self.pet.bed_pos["x"] - 20, self.pet.bed_pos["y"] - 12, 12, 10))

        # Tub Details
        pygame.draw.rect(self.screen, (223, 230, 233), (self.pet.wash_pos["x"] - 22, self.pet.wash_pos["y"] - 12, 44, 24))
        pygame.draw.rect(self.screen, (9, 132, 227), (self.pet.wash_pos["x"] - 18, self.pet.wash_pos["y"] - 8, 36, 16))

        # Toy Chest Details
        pygame.draw.rect(self.screen, (214, 48, 49), (self.pet.chest_pos["x"] - 20, self.pet.chest_pos["y"] - 12, 40, 24))
        pygame.draw.rect(self.screen, (253, 203, 110), (self.pet.chest_pos["x"] - 20, self.pet.chest_pos["y"] - 12, 40, 4))

        # Draw Toy if placed and not carried
        if self.pet.toy_placed:
            draw_object_base(self.pet.toy_pos, 22, "장난감 (Toy)")
            if not self.pet.carrying_toy:
                pygame.draw.circle(self.screen, COLOR_BORED, (int(self.pet.toy_pos["x"]), int(self.pet.toy_pos["y"])), 8)
                # Stripe
                pygame.draw.circle(self.screen, COLOR_STAMINA, (int(self.pet.toy_pos["x"]), int(self.pet.toy_pos["y"])), 4)

        # Draw Pet Shadow & Body (Using visual draw_x and draw_y)
        pygame.draw.ellipse(self.screen, SHADOW, (self.draw_x - 16, self.draw_y + 8, 32, 10))
        pet_color = (255, 234, 167) if self.pet.carrying_toy else ACCENT
        pygame.draw.circle(self.screen, pet_color, (int(self.draw_x), int(self.draw_y)), 16)
        pygame.draw.circle(self.screen, (255, 255, 255, 50), (int(self.draw_x), int(self.draw_y)), 16, width=2)

        # Draw Face expressions (Using visual draw_x and draw_y)
        is_sleeping = self.pet.last_action in [self.pet.ACTIONS["SLEEP_BED"], self.pet.ACTIONS["SLEEP_FLOOR"]]
        is_bored = self.pet.boredom > 0.7
        
        if is_sleeping:
            # Sleep closed eyes 'u u'
            lbl_eye = self.font_body_bold.render("u u", True, BG_DARK)
            self.screen.blit(lbl_eye, (int(self.draw_x) - lbl_eye.get_width() // 2, int(self.draw_y) - 6))
        elif is_bored:
            # Bored/angry face
            lbl_eye = self.font_small.render("ò  ó", True, BG_DARK)
            self.screen.blit(lbl_eye, (int(self.draw_x) - lbl_eye.get_width() // 2, int(self.draw_y) - 8))
            pygame.draw.arc(self.screen, BG_DARK, (int(self.draw_x) - 4, int(self.draw_y) + 2, 8, 6), 0, math.pi, 2)
        else:
            # Smile face
            pygame.draw.circle(self.screen, BG_DARK, (int(self.draw_x) - 5, int(self.draw_y) - 2), 2)
            pygame.draw.circle(self.screen, BG_DARK, (int(self.draw_x) + 5, int(self.draw_y) - 2), 2)
            pygame.draw.arc(self.screen, BG_DARK, (int(self.draw_x) - 4, int(self.draw_y) - 2, 8, 8), math.pi, 0, 2)

        # Draw toy if carried by pet (Using visual draw_x and draw_y)
        if self.pet.carrying_toy:
            pygame.draw.circle(self.screen, COLOR_BORED, (int(self.draw_x) + 10, int(self.draw_y) + 10), 6)

        # Draw Particles
        for p in self.particles:
            surf = pygame.Surface((int(p["radius"] * 2), int(p["radius"] * 2)), pygame.SRCALPHA)
            alpha_val = int(p["alpha"] * 255)
            if p["type"] == 'sleep':
                # Draw Z
                font_z = self.font_small.render('Z', True, p["color"])
                surf.blit(font_z, (0, 0))
            else:
                pygame.draw.circle(surf, (*p["color"], alpha_val), (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(surf, (p["x"] - p["radius"], p["y"] - p["radius"]))

        # Action Overlay Banner at the bottom of the room
        banner_rect = pygame.Rect(12, 345, 576, 42)
        # Translucent banner background
        banner_surf = pygame.Surface((576, 42), pygame.SRCALPHA)
        banner_surf.fill((11, 13, 25, 210))
        self.screen.blit(banner_surf, banner_rect.topleft)
        pygame.draw.rect(self.screen, PANEL_BORDER, banner_rect, width=1, border_radius=6)

        status_txt = self.font_body_bold.render(self.pet.last_action_status, True, TEXT_PRIMARY)
        self.screen.blit(status_txt, (24, 356))

        rew_color = COLOR_CLEAN if self.pet.last_reward >= 0 else DANGER
        sign = "+" if self.pet.last_reward >= 0 else ""
        rew_txt = self.font_title.render(f"{sign}{self.pet.last_reward:.3f}", True, rew_color)
        self.screen.blit(rew_txt, (588 - rew_txt.get_width() - 12, 352))

        # Draw Control Panel Y: 400 to 600
        ctrl_bg = pygame.Rect(0, 400, 600, 200)
        pygame.draw.rect(self.screen, BG_DARK, ctrl_bg)
        pygame.draw.line(self.screen, PANEL_BORDER, (0, 400), (600, 400), 2)

        # Toggle Button state based on simulation
        self.btn_pause.text = "시뮬레이션 시작" if not is_running else "시뮬레이션 일시정지"
        self.btn_pause.bg_color = SUCCESS if not is_running else ACCENT
        
        # Disabled states
        self.btn_clean.is_disabled = not self.pet.toy_placed
        self.btn_toy.is_disabled = self.pet.toy_placed

        # Draw Buttons
        self.btn_clean.draw(self.screen, self.font_body_bold)
        self.btn_toy.draw(self.screen, self.font_body_bold)
        self.btn_pretrain.draw(self.screen, self.font_body_bold)
        self.btn_pause.draw(self.screen, self.font_body_bold)
        self.btn_reset.draw(self.screen, self.font_body_bold)
        self.speed_slider.draw(self.screen, self.font_small, self.font_small)

        # Speed slider text description
        speed_lbl = self.font_body_bold.render(f"{int(self.speed_slider.current_val)}x", True, TEXT_PRIMARY)
        self.screen.blit(speed_lbl, (588 - speed_lbl.get_width(), 512))

        # Status text logger description
        pygame.draw.rect(self.screen, (16, 20, 38), (20, 525, 360, 50), border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, (20, 525, 360, 50), width=1, border_radius=6)
        
        cmd_title = self.font_small.render("주인의 명령 센터 로그", True, (162, 155, 254))
        self.screen.blit(cmd_title, (30, 532))
        cmd_desc = self.font_small.render(self.status_log, True, self.status_log_color)
        self.screen.blit(cmd_desc, (30, 552))

        # 3. Draw Right Side Panel (Control and Statistics)
        right_panel = pygame.Rect(600, 0, 400, 600)
        pygame.draw.rect(self.screen, PANEL_BG, right_panel)
        pygame.draw.line(self.screen, PANEL_BORDER, (600, 0), (600, 600), 2)

        # Header Stats Y: 10 to 60
        title_surf = self.font_section.render("DRL 펫 성격 시뮬레이터", True, COLOR_CLEAN)
        self.screen.blit(title_surf, (620, 15))

        # Header stats pills
        eps_surf = self.font_small.render(f"에피소드: {episode_count}  |  걸음수: {step_count}  |  Epsilon: {self.agent.epsilon:.2f}", True, TEXT_SECONDARY)
        self.screen.blit(eps_surf, (620, 42))

        # Draw sliders Y: 70 to 300
        for slider in self.sliders.values():
            slider.draw(self.screen, self.font_small, self.font_small)

        # Status progress bars Y: 310 to 390 (2x2 Layout)
        bar_lbl_surf = self.font_section.render("펫의 현재 상태 (Real-time Status)", True, TEXT_PRIMARY)
        self.screen.blit(bar_lbl_surf, (620, 298))

        def draw_status_bar(x, y, label, val, color):
            lbl = self.font_small.render(f"{label} ({int(val*100)}%)", True, TEXT_SECONDARY)
            self.screen.blit(lbl, (x, y))
            # Background track
            pygame.draw.rect(self.screen, (35, 39, 66), (x, y + 16, 160, 10), border_radius=4)
            # Fill track
            fill_w = int(val * 160)
            if fill_w > 0:
                pygame.draw.rect(self.screen, color, (x, y + 16, fill_w, 10), border_radius=4)

        draw_status_bar(620, 325, "청결도", self.pet.cleanliness, COLOR_CLEAN)
        draw_status_bar(800, 325, "포만감", self.pet.fullness, COLOR_FULL)
        draw_status_bar(620, 362, "스태미너", self.pet.stamina, COLOR_STAMINA)
        draw_status_bar(800, 362, "지루함", self.pet.boredom, COLOR_BORED)

        # DQN Action Q-values Y: 398 to 515 (Small horizontal bar charts with selection percentage)
        q_lbl_surf = self.font_section.render("DQN 분석 (선택 비율 | Q값)", True, TEXT_PRIMARY)
        self.screen.blit(q_lbl_surf, (620, 398))

        s = self.pet.get_state_vector()
        q_values = self.agent.get_q_values(s)
        max_q_idx = np.argmax(q_values)

        q_bar_y = 418
        q_bar_h = 8
        q_bar_max_w = 110
        q_spacing = 11

        total_steps = self.pet.total_action_steps

        for i in range(self.agent.action_size):
            q_val = q_values[i]
            
            # Calculate action percentage
            act_count = self.pet.action_counts[i]
            pct = (act_count / total_steps * 100.0) if total_steps > 0 else 0.0
            
            # Print Action label
            lbl_act = self.font_small.render(self.action_names[i], True, TEXT_PRIMARY if i == max_q_idx else TEXT_SECONDARY)
            self.screen.blit(lbl_act, (620, q_bar_y + i * q_spacing))
            
            # Print Percentage label
            pct_color = COLOR_CLEAN if i == max_q_idx else TEXT_SECONDARY
            lbl_pct = self.font_small.render(f"{pct:.1f}%", True, pct_color)
            self.screen.blit(lbl_pct, (742, q_bar_y + i * q_spacing))
            
            # Draw Q-bar
            # Normalize q_value (map -1.0 to 1.0 -> 0 to q_bar_max_w)
            norm_q = max(0.0, min(1.0, (q_val + 1.0) / 2.0))
            bar_w = int(norm_q * q_bar_max_w)
            
            bar_color = ACCENT if i == max_q_idx else (80, 85, 115)
            # Track
            pygame.draw.rect(self.screen, (35, 39, 66), (785, q_bar_y + i * q_spacing + 2, q_bar_max_w, q_bar_h), border_radius=4)
            # Fill
            if bar_w > 0:
                pygame.draw.rect(self.screen, bar_color, (785, q_bar_y + i * q_spacing + 2, bar_w, q_bar_h), border_radius=4)

            # Value label
            lbl_q = self.font_small.render(f"{q_val:.2f}", True, TEXT_PRIMARY if i == max_q_idx else TEXT_SECONDARY)
            self.screen.blit(lbl_q, (910, q_bar_y + i * q_spacing))

        # Reward Plot Canvas Y: 512 to 595 (Height 80)
        # Draw frame
        plot_rect = pygame.Rect(620, 520, 360, 68)
        pygame.draw.rect(self.screen, (15, 18, 32), plot_rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, plot_rect, width=1, border_radius=6)

        # Plot Reward line
        history = list(self.agent.recent_losses)  # We can draw loss or rewards. Let's pass episode rewards from main
        # Wait, let's draw the history list that main will provide.
        # We will set main to update a history list on the simulator instance
        
    def draw_reward_history(self, reward_history):
        if len(reward_history) < 2:
            lbl_no_data = self.font_small.render("학습 결과가 쌓이면 그래프가 활성화됩니다.", True, TEXT_SECONDARY)
            self.screen.blit(lbl_no_data, (800 - lbl_no_data.get_width() // 2, 545))
            return

        w = 340
        h = 52
        x_start = 630
        y_start = 528

        # Min / Max values
        min_v = -0.3
        max_v = 0.5
        v_range = max_v - min_v

        points = []
        for i, val in enumerate(reward_history):
            pct_x = i / (len(reward_history) - 1)
            px = x_start + int(pct_x * w)
            
            # Normalize to plot height
            norm_y = (val - min_v) / v_range
            norm_y = max(0.0, min(1.0, norm_y))
            py = y_start + h - int(norm_y * h)
            points.append((px, py))

        pygame.draw.lines(self.screen, COLOR_CLEAN, False, points, 2)

    def handle_events(self):
        actions_triggered = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions_triggered.append("cmd_quit")

            # Handle Slider events
            for key, slider in self.sliders.items():
                if slider.handle_event(event):
                    # Notify main to update pet personality
                    actions_triggered.append(("personality_change", key, slider.current_val))

            # Speed Slider
            if self.speed_slider.handle_event(event):
                actions_triggered.append(("speed_change", self.speed_slider.current_val))

            # Buttons
            if self.btn_clean.handle_event(event):
                actions_triggered.append("cmd_clean")
            if self.btn_toy.handle_event(event):
                actions_triggered.append("cmd_place_toy")
            if self.btn_pretrain.handle_event(event):
                actions_triggered.append("cmd_pretrain")
            if self.btn_pause.handle_event(event):
                actions_triggered.append("cmd_toggle_pause")
            if self.btn_reset.handle_event(event):
                actions_triggered.append("cmd_reset")

        return actions_triggered
