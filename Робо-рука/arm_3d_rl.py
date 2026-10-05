"""
arm_3d_rl.py — 3D-симуляция робо-руки (Ursina) с обучением DQN-агента.

Манипулятор имеет 3 сустава:
  * yaw      — поворот базы вокруг вертикальной оси Y;
  * shoulder — наклон плеча (вращение вокруг оси X);
  * elbow    — сгиб локтя (вращение вокруг оси X).

Агент наблюдает углы суставов и вектор до цели, а действует малыми шагами
по каждому суставу. Схват должен попасть в случайную цель в рабочем объёме.

Управление:
  [SPACE] — Turbo-режим (несколько шагов обучения за кадр)
  [S]     — сохранить веса модели в arm_model_3d.pth
  ПКМ + мышь / колесо (EditorCamera) — свободная камера

Требуется: pip install ursina torch
"""
import math
import random
import sys
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from ursina import (Ursina, Entity, DirectionalLight, EditorCamera, Text,
                        Vec3, color, window)
except ImportError:
    sys.exit("Не найден пакет ursina. Установите его командой: pip install ursina")

# ==================== НАСТРОЙКИ СРЕДЫ ====================
L1, L2 = 2.0, 1.6                      # Длины звеньев, м
BASE_HEIGHT = 0.8                      # Высота плечевого узла над полом, м
STEP = 5.0                             # Шаг действия агента, градусов
JOINT_LIMIT = (-89.0, 89.0)            # Ограничения плеча и локтя, градусов
REACH_TOLERANCE = 0.25                 # Порог достижения цели, м
MIN_REACH = 1.0                        # Минимальный радиус случайной цели, м
MAX_REACH = L1 + L2 - 0.3              # Максимальный радиус случайной цели, м

ACTIONS = [                            # (delta_yaw, delta_shoulder, delta_elbow)
    (STEP, 0, 0), (-STEP, 0, 0),
    (0, STEP, 0), (0, -STEP, 0),
    (0, 0, STEP), (0, 0, -STEP),
]
# sin/cos yaw + sin/cos shoulder + вектор до цели (3 компоненты) = 7
STATE_DIM = 4 + 3
ACTION_DIM = len(ACTIONS)

MODEL_PATH = "arm_model_3d.pth"


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def end_effector_position(yaw, shoulder, elbow):
    """Forward kinematics 3DOF-манипулятора (совпадает с иерархией Entity)."""
    a1 = math.radians(shoulder)
    a2 = math.radians(shoulder + elbow)
    r_horizontal = L1 * math.sin(a1) + L2 * math.sin(a2)
    height = BASE_HEIGHT + L1 * math.cos(a1) + L2 * math.cos(a2)
    y_rad = math.radians(yaw)
    return Vec3(r_horizontal * math.sin(y_rad), height, r_horizontal * math.cos(y_rad))


def get_random_target():
    """Случайная достижимая цель внутри рабочего полусферического объёма."""
    angle = random.uniform(0, 2 * math.pi)
    radius = random.uniform(MIN_REACH, MAX_REACH)
    height = random.uniform(0.5, BASE_HEIGHT + L1 + L2 - 0.3)
    return Vec3(radius * math.cos(angle), height, radius * math.sin(angle))


# ==================== НЕЙРОСЕТЬ (DQN) ====================
class DQN(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
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
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=20000)

        self.gamma = 0.98
        self.epsilon = 1.0             # На старте агент исследует мир случайно
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997     # Постепенно переходит к опыту нейросети
        self.batch_size = 64

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(ACTION_DIM)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(torch.argmax(self.policy_net(s)))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.tensor(states, dtype=torch.float32).to(self.device)
        a = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        r = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        ns = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        d = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_current = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            max_next_q = self.target_net(ns).max(1)[0].unsqueeze(1)
            q_target = r + (1 - d) * self.gamma * max_next_q

        loss = nn.MSELoss()(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # Ограничение нормы градиента — защита от "взрыва" Q-значений
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ==================== 3D-СЦЕНА ====================
app = Ursina()
window.title = "3D Robotic Arm RL (DQN)"
EditorCamera()

Entity(model='plane', scale=30, color=color.dark_gray,
       texture='white_cube', texture_scale=(30, 30))
DirectionalLight(y=10, z=-10, shadows=True)

# Иерархия манипулятора
base_stand = Entity(model='cylinder', color=color.gray, scale=(1.2, 0.4, 1.2), y=0.2)
base_yaw = Entity(parent=base_stand, model='cube', color=color.azure,
                  scale=(0.8, 0.6, 0.8), y=0.45)

shoulder_pivot = Entity(parent=base_yaw, y=0.3)
shoulder_link = Entity(parent=shoulder_pivot, model='cube', color=color.cyan,
                       scale=(0.3, L1, 0.3), y=L1 / 2)

elbow_pivot = Entity(parent=shoulder_pivot, y=L1)
elbow_link = Entity(parent=elbow_pivot, model='cube', color=color.orange,
                    scale=(0.25, L2, 0.25), y=L2 / 2)

wrist = Entity(parent=elbow_pivot, y=L2)
end_effector = Entity(parent=wrist, model='sphere', color=color.yellow, scale=0.3)

target_entity = Entity(model='sphere', color=color.red, scale=0.35)

info_text = Text(position=(-0.9, 0.45), scale=1.2, color=color.yellow)
Text(position=(-0.9, -0.45), scale=1.0,
     text="[SPACE] - Turbo | [S] - Сохранить модель | ПКМ+мышь - камера")

# ==================== СОСТОЯНИЕ ОБУЧЕНИЯ ====================
agent = ArmAgent()
print(f"Обучение запущено на: {agent.device}")

theta_yaw = 0.0
theta_shoulder = 30.0
theta_elbow = -45.0
target_pt = get_random_target()

episode = 1
steps = 0
max_steps_per_episode = 250
hits = 0
turbo = False
loss_ema = None


def get_state(end_pt, tgt_pt):
    dist = tgt_pt - end_pt
    norm = L1 + L2
    return [
        math.sin(math.radians(theta_yaw)),
        math.cos(math.radians(theta_yaw)),
        math.sin(math.radians(theta_shoulder)),
        math.cos(math.radians(theta_shoulder)),
        dist.x / norm, dist.y / norm, dist.z / norm,
    ]


def reset_episode():
    """Случайная стартовая конфигурация и новая цель."""
    global theta_yaw, theta_shoulder, theta_elbow, target_pt, steps
    theta_yaw = random.uniform(-180, 180)
    theta_shoulder = random.uniform(10, 70)
    theta_elbow = random.uniform(-80, 0)
    target_pt = get_random_target()
    steps = 0


def step_once():
    """Один шаг среды: действие -> награда -> обучение -> синхронизация сцены."""
    global episode, hits, steps, loss_ema

    end_pt = end_effector_position(theta_yaw, theta_shoulder, theta_elbow)
    state = get_state(end_pt, target_pt)
    dist_before = (target_pt - end_pt).length()

    action = agent.select_action(state)
    d_yaw, d_sh, d_el = ACTIONS[action]
    theta_yaw = (theta_yaw + d_yaw) % 360.0
    theta_shoulder = clamp(theta_shoulder + d_sh, *JOINT_LIMIT)
    theta_elbow = clamp(theta_elbow + d_el, *JOINT_LIMIT)

    new_end = end_effector_position(theta_yaw, theta_shoulder, theta_elbow)
    dist_after = (target_pt - new_end).length()

    # Функция наград: прогресс по дистанции + штраф за шаг + бонус за попадание
    reward = (dist_before - dist_after)
    reward -= 0.01
    done = False
    if dist_after < REACH_TOLERANCE:
        reward += 20.0
        hits += 1
        done = True

    steps += 1
    if steps >= max_steps_per_episode:
        done = True

    next_state = get_state(new_end, target_pt)
    agent.memory.append((state, action, reward, next_state, done))
    loss = agent.train_step()
    if loss is not None:
        loss_ema = loss if loss_ema is None else 0.95 * loss_ema + 0.05 * loss

    if done:
        agent.update_epsilon()
        if episode % 5 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        episode += 1
        reset_episode()
        new_end = end_effector_position(theta_yaw, theta_shoulder, theta_elbow)
        dist_after = (target_pt - new_end).length()

    # Синхронизация 3D-сцены с состоянием суставов
    base_yaw.rotation_y = theta_yaw
    shoulder_pivot.rotation_x = -theta_shoulder
    elbow_pivot.rotation_x = -theta_elbow
    target_entity.position = target_pt

    reached = dist_after < REACH_TOLERANCE
    end_effector.color = color.lime if reached else color.yellow
    target_entity.color = color.green if reached else color.red

    loss_str = f"{loss_ema:.4f}" if loss_ema is not None else "накопление опыта..."
    info_text.text = (
        f"Эпизод: {episode}\n"
        f"Попаданий: {hits}\n"
        f"Дистанция: {dist_after:.2f} м\n"
        f"Epsilon: {agent.epsilon:.3f}\n"
        f"Loss (EMA): {loss_str}\n"
        f"Turbo: {'ВКЛ' if turbo else 'выкл'}\n"
        f"Устройство: {agent.device}"
    )


def input(key):
    global turbo
    if key == 'space':
        turbo = not turbo
    elif key == 's':
        torch.save(agent.policy_net.state_dict(), MODEL_PATH)
        print(f"Модель сохранена в {MODEL_PATH}! 🎉")


def update():
    # В турбо-режиме прогоняем несколько шагов обучения за кадр
    iterations = 10 if turbo else 1
    for _ in range(iterations):
        step_once()


reset_episode()
app.run()
