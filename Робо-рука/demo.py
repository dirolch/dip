import pygame
import math
import sys
import torch
import torch.nn as nn

# ==================== НАСТРОЙКИ СРЕДЫ ====================
WIDTH, HEIGHT = 800, 600
BASE_X, BASE_Y = WIDTH // 2, HEIGHT - 100
L1, L2 = 150, 120

ACTIONS = [(0.06, 0), (-0.06, 0), (0, 0.06), (0, -0.06)]

def forward_kinematics(t1, t2):
    j_x = BASE_X + L1 * math.cos(t1)
    j_y = BASE_Y - L1 * math.sin(t1)
    e_x = j_x + L2 * math.cos(t1 + t2)
    e_y = j_y - L2 * math.sin(t1 + t2)
    return (j_x, j_y), (e_x, e_y)

def get_state(t1, t2, end_pt, tgt_pt):
    return [
        math.sin(t1),
        math.cos(t1),
        math.sin(t2),
        math.cos(t2),
        (tgt_pt[0] - end_pt[0]) / (L1 + L2),
        (tgt_pt[1] - end_pt[1]) / (L1 + L2)
    ]

# ==================== НЕЙРОСЕТЬ ====================
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

# Загрузка обученных весов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN().to(device)

try:
    model.load_state_dict(torch.load("arm_model.pth", map_location=device))
    model.eval()
    print("Веса модели arm_model.pth успешно загружены! 🚀")
except FileNotFoundError:
    print("Ошибка: Файл arm_model.pth не найден! Сначала сохрани модель в arm_rl.py клавишей S.")
    sys.exit()

# ==================== ДЕМО-РЕЖИМ (ИНФЕРЕНС) ====================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robotic Arm – Demo Mode (Click to set target)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

theta1 = math.radians(90)
theta2 = math.radians(0)
target_x, target_y = BASE_X, BASE_Y - 180

running = True

while running:
    clock.tick(60)
    screen.fill((25, 25, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Клик левой кнопкой мыши
                mx, my = pygame.mouse.get_pos()
                # Проверяем, чтобы точка была в зоне досягаемости полукруга
                dx = mx - BASE_X
                dy = BASE_Y - my
                dist = math.hypot(dx, dy)
                if dist <= (L1 + L2) and dy >= 0:
                    target_x, target_y = mx, my

    # Опрос состояния
    (joint_x, joint_y), (end_x, end_y) = forward_kinematics(theta1, theta2)
    state = get_state(theta1, theta2, (end_x, end_y), (target_x, target_y))
    dist = math.hypot(target_x - end_x, target_y - end_y)

    # Чистый выбор нейросети без случайности
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_vals = model(s)
        action = torch.argmax(q_vals).item()

    d_t1, d_t2 = ACTIONS[action]
    theta1 = (theta1 + d_t1) % (2 * math.pi)
    theta2 = (theta2 + d_t2) % (2 * math.pi)

        # Отрисовка
    pygame.draw.circle(screen, (40, 40, 50), (BASE_X, BASE_Y), L1 + L2, 1)
    pygame.draw.circle(screen, (240, 70, 70), (int(target_x), int(target_y)), 10)
    pygame.draw.rect(screen, (80, 80, 90), (BASE_X - 35, BASE_Y, 70, 16), border_radius=4)
    pygame.draw.line(screen, (70, 140, 240), (BASE_X, BASE_Y), (joint_x, joint_y), 8)
    pygame.draw.line(screen, (90, 200, 250), (joint_x, joint_y), (end_x, end_y), 6)
    pygame.draw.circle(screen, (220, 220, 230), (BASE_X, BASE_Y), 8)
    pygame.draw.circle(screen, (220, 220, 230), (int(joint_x), int(joint_y)), 6)
    pygame.draw.circle(screen, (50, 220, 100) if dist < 20 else (250, 180, 50), (int(end_x), int(end_y)), 6)

    # Телеметрия инференса
    info = [
        "Режим: Тестирование обученной модели (Инференс)",
        "Кликни ЛКМ в полукруг, чтобы задать цель",
        f"Дистанция до цели: {dist:.1f} px",
        f"Устройство: {device}"
    ]
    for i, txt in enumerate(info):
        render = font.render(txt, True, (210, 210, 220))
        screen.blit(render, (20, 20 + i * 22))

    pygame.display.flip()

pygame.quit()
sys.exit()
