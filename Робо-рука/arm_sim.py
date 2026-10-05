import pygame
import math
import sys

# Инициализация Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2-Link Robotic Arm Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# Константы базы манипулятора
BASE_X, BASE_Y = WIDTH // 2, HEIGHT - 100

# Длины звеньев (константы)
L1 = 160  # Первое звено (плечо)
L2 = 130  # Второе звено (предплечье)

# Ограничения суставов (в радианах), чтобы рука не уходила под пол
MIN_ANGLE = math.radians(-10)
MAX_ANGLE = math.radians(190)

# Константы управления
SPEED = 0.04  # Скорость поворота суставов (рад/кадр)
REACH_TOLERANCE = 15  # Порог "достижения" цели, px

# Позиция цели
target_x, target_y = 550, 250

def forward_kinematics(theta1, theta2):
    """Вычисляет координаты сочленений по углам (Forward Kinematics)"""
    # Координата сустава 1 (локоть)
    joint_x = BASE_X + L1 * math.cos(theta1)
    joint_y = BASE_Y - L1 * math.sin(theta1)
    
    # Координата рабочего органа (кисть / схват)
    end_x = joint_x + L2 * math.cos(theta1 + theta2)
    end_y = joint_y - L2 * math.sin(theta1 + theta2)
    
    return (joint_x, joint_y), (end_x, end_y)

def clamp_angle(angle):
    """Ограничивает угол сустава физическими пределами."""
    return max(MIN_ANGLE, min(MAX_ANGLE, angle))


running = True
theta1 = math.radians(45)  # Угол первого сустава
theta2 = math.radians(45)  # Угол второго сустава относительно первого
while running:
    clock.tick(60)
    screen.fill((30, 30, 35))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Цель можно ставить только в пределах рабочей зоны (полукруг над полом)
            dx, dy = event.pos[0] - BASE_X, BASE_Y - event.pos[1]
            if dy >= 0 and math.hypot(dx, dy) <= L1 + L2:
                target_x, target_y = event.pos

    # Управление с клавиатуры (зажатые клавиши обрабатываются каждый кадр)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_r]:
        theta1, theta2 = math.radians(45), math.radians(45)
    else:
        if keys[pygame.K_a]:
            theta1 = clamp_angle(theta1 + SPEED)
        if keys[pygame.K_d]:
            theta1 = clamp_angle(theta1 - SPEED)
        if keys[pygame.K_w]:
            theta2 = clamp_angle(theta2 + SPEED)
        if keys[pygame.K_s]:
            theta2 = clamp_angle(theta2 - SPEED)

    # Расчёт геометрии
    (joint_x, joint_y), (end_x, end_y) = forward_kinematics(theta1, theta2)

    # Расстояние от схвата до цели
    dist = math.hypot(target_x - end_x, target_y - end_y)

    # Рисуем рабочую зону
    pygame.draw.circle(screen, (45, 45, 55), (BASE_X, BASE_Y), L1 + L2, 1)
    pygame.draw.line(screen, (60, 60, 70), (0, BASE_Y + 20), (WIDTH, BASE_Y + 20), 2)  # Пол

    # Рисуем цель
    pygame.draw.circle(screen, (230, 60, 60), (int(target_x), int(target_y)), 12)
    pygame.draw.circle(screen, (255, 255, 255), (int(target_x), int(target_y)), 4)

    # Рисуем звенья руки
    pygame.draw.rect(screen, (100, 100, 110), (BASE_X - 40, BASE_Y, 80, 20), border_radius=4)  # Основание
    pygame.draw.line(screen, (70, 140, 240), (BASE_X, BASE_Y), (joint_x, joint_y), 10)  # Звено 1
    pygame.draw.line(screen, (90, 200, 250), (joint_x, joint_y), (end_x, end_y), 7)  # Звено 2

    # Суставы (шарниры)
    pygame.draw.circle(screen, (220, 220, 230), (BASE_X, BASE_Y), 10)
    pygame.draw.circle(screen, (220, 220, 230), (int(joint_x), int(joint_y)), 8)

    # Схват (End-Effector)
    color = (50, 220, 100) if dist < REACH_TOLERANCE else (240, 180, 50)
    pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 7)

    # Линия от схвата к цели (визуальная подсказка)
    pygame.draw.line(screen, (120, 120, 130), (int(end_x), int(end_y)),
                     (int(target_x), int(target_y)), 1)

    # Текстовая телеметрия
    info_lines = [
        "Управление:",
        "  [A / D] - Сустав 1 (Плечо)",
        "  [W / S] - Сустав 2 (Локоть)",
        "  [ЛКМ]   - Переместить цель",
        "  [R]     - Сброс углов",
        "",
        f"Угол 1: {math.degrees(theta1):.1f}°",
        f"Угол 2: {math.degrees(theta2):.1f}°",
        f"Дистанция до цели: {dist:.1f} px",
        "Статус: " + ("ДОСТИГНУТА! 🎯" if dist < REACH_TOLERANCE else "В процессе...")
    ]

    for i, line in enumerate(info_lines):
        txt = font.render(line, True, (200, 200, 210))
        screen.blit(txt, (20, 20 + i * 24))

    pygame.display.flip()

pygame.quit()
sys.exit()

