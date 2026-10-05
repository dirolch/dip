import pygame
import math
import random
import sys
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# ==================== НАСТРОЙКИ СРЕДЫ ====================
WIDTH, HEIGHT = 800, 600
BASE_X, BASE_Y = WIDTH // 2, HEIGHT - 100
L1, L2 = 150, 120

# Пространство действий: 4 возможных движения
# 0: Joint1 +, 1: Joint1 -, 2: Joint2 +, 3: Joint2 -
ACTIONS = [(0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)]

def forward_kinematics(t1, t2):
    j_x = BASE_X + L1 * math.cos(t1)
    j_y = BASE_Y - L1 * math.sin(t1)
    e_x = j_x + L2 * math.cos(t1 + t2)
    e_y = j_y - L2 * math.sin(t1 + t2)
    return (j_x, j_y), (e_x, e_y)

def get_random_target():
    angle = random.uniform(math.radians(20), math.radians(160))
    radius = random.uniform(80, L1 + L2 - 20)
    tx = BASE_X + radius * math.cos(angle)
    ty = BASE_Y - radius * math.sin(angle)
    return tx, ty

# ==================== НЕЙРОСЕТЬ (DQN) ====================
class DQN(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==================== RL-АГЕНТ ====================
class ArmAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=20000)

        self.gamma = 0.98
        self.epsilon = 1.0        # На старте агент исследует мир случайно
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998 # Постепенно переходит к опыту нейросети
        self.batch_size = 64

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS) - 1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(s)
            return torch.argmax(q_vals).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_current = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            max_next_q = self.target_net(ns).max(1)[0].unsqueeze(1)
            q_target = r + (1 - d) * self.gamma * max_next_q

        loss = nn.MSELoss()(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ==================== СИМУЛЯЦИЯ ====================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robotic Arm Reinforcement Learning (DQN)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

agent = ArmAgent()
print(f"Обучение запущено на: {agent.device}")

theta1 = math.radians(90)
theta2 = math.radians(0)
target_x, target_y = get_random_target()

episode = 1
steps = 0
max_steps_per_episode = 200
score = 0
hits = 0

def get_state(t1, t2, end_pt, tgt_pt):
    return [
        math.sin(t1),
        math.cos(t1),
        math.sin(t2),
        math.cos(t2),
        (tgt_pt[0] - end_pt[0]) / (L1 + L2),
        (tgt_pt[1] - end_pt[1]) / (L1 + L2)
    ]

running = True
target_update_freq = 5
turbo = False

while running:
    # Включаем / выключаем ограничение FPS по клавише SPACE
    # Если turbo = True – симуляция крутится на максимальной скорости
    if not turbo:
        clock.tick(60)

    screen.fill((25, 25, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                turbo = not turbo  # Переключение Турбо-режима
            elif event.key == pygame.K_s:
                torch.save(agent.policy_net.state_dict(), "arm_model.pth")
                print("Модель успешно сохранена в arm_model.pth! 🎉")




    _, end_pos = forward_kinematics(theta1, theta2)
    state = get_state(theta1, theta2, end_pos, (target_x, target_y))
    dist_before = math.hypot(target_x - end_pos[0], target_y - end_pos[1])

    # Выбор действия
    action = agent.select_action(state)
    d_t1, d_t2 = ACTIONS[action]
    theta1 = (theta1 + d_t1) % (2 * math.pi)
    theta2 = (theta2 + d_t2) % (2 * math.pi)

    (joint_x, joint_y), (end_x, end_y) = forward_kinematics(theta1, theta2)
    dist_after = math.hypot(target_x - end_x, target_y - end_y)

    # Функция наград
    reward = (dist_before - dist_after) * 0.1
    reward -= 0.01
    done = False

    if dist_after < 20:
        reward += 20.0
        hits += 1
        done = True

    steps += 1
    if steps >= max_steps_per_episode:
        done = True

    next_state = get_state(theta1, theta2, (end_x, end_y), (target_x, target_y))
    agent.memory.append((state, action, reward, next_state, done))
    agent.train_step()
    score += reward

    if done:
        agent.update_epsilon()
        if episode % target_update_freq == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        episode += 1
        steps = 0
        score = 0
        target_x, target_y = get_random_target()

    # Отрисовка
    pygame.draw.circle(screen, (40, 40, 50), (BASE_X, BASE_Y), L1 + L2, 1)
    pygame.draw.circle(screen, (240, 70, 70), (int(target_x), int(target_y)), 10)
    pygame.draw.rect(screen, (80, 80, 90), (BASE_X - 35, BASE_Y, 70, 16), border_radius=4)
    pygame.draw.line(screen, (70, 140, 240), (BASE_X, BASE_Y), (joint_x, joint_y), 8)
    pygame.draw.line(screen, (90, 200, 250), (joint_x, joint_y), (end_x, end_y), 6)
    pygame.draw.circle(screen, (220, 220, 230), (BASE_X, BASE_Y), 8)
    pygame.draw.circle(screen, (220, 220, 230), (int(joint_x), int(joint_y)), 6)
    pygame.draw.circle(screen, (50, 220, 100) if dist_after < 20 else (250, 180, 50), (int(end_x), int(end_y)), 6)

    # Телеметрия обучения
    info = [
        f"Эпизод: {episode}",
        f"Успешных попаданий: {hits}",
        f"Дистанция: {dist_after:.1f} px",
        f"Epsilon (исследование): {agent.epsilon:.3f}",
        f"Устройство: {agent.device}"
    ]
    for i, txt in enumerate(info):
        render = font.render(txt, True, (210, 210, 220))
        screen.blit(render, (20, 20 + i * 22))

    pygame.display.flip()

pygame.quit()
sys.exit()
