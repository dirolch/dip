from ursina import *
import math

app = Ursina()

# ==================== НАСТРОЙКИ СЦЕНЫ ====================
window.title = "3D Robotic Arm - Test Environment"
window.borderless = False
window.size = (1000, 700)

# Свободная 3D-камера (зажми правую кнопку мыши, чтобы вращать, колесо – зум)
EditorCamera()

# Пол с сеткой
grid = Entity(model='plane', scale=20, color=color.dark_gray, texture='white_cube', texture_scale=(20, 20))
DirectionalLight(y=10, z=-10, shadows=True)

# ==================== ЗВЕНЬЯ МАНИПУЛЯТОРА ====================
# Длины звеньев в метрах
L1 = 2.0  # Плечо
L2 = 1.6  # Предплечье

# 1. Базовая тумба
base_stand = Entity(model='cylinder', color=color.gray, scale=(1.2, 0.4, 1.2), y=0.2)

# 2. Поворотная башня (Сустав 0 - вращение вокруг оси Y)
base_yaw = Entity(model='cube', color=color.azure, scale=(0.8, 0.6, 0.8), y=0.5)

# 3. Первое звено (Сустав 1 - наклон плеча)
shoulder_pivot = Entity(parent=base_yaw, y=0.3)
shoulder_link = Entity(parent=shoulder_pivot, model='cube', color=color.cyan, scale=(0.3, L1, 0.3), y=L1/2)

# 4. Второе звено (Сустав 2 - сгиб локтя)
elbow_pivot = Entity(parent=shoulder_pivot, y=L1)
elbow_link = Entity(parent=elbow_pivot, model='cube', color=color.orange, scale=(0.25, L2, 0.25), y=L2/2)

# 5. Схват / Наконечник (обязательно parent=wrist, иначе он не двигается с рукой)
wrist = Entity(parent=elbow_pivot, y=L2)
end_effector = Entity(parent=wrist, model='sphere', color=color.yellow, scale=0.3)

# 6. Целевая точка (красная сфера)
target = Entity(model='sphere', color=color.red, scale=0.35, position=(1.5, 2.0, 1.0))

# ==================== УПРАВЛЕНИЕ И ОБНОВЛЕНИЕ ====================
# Углы суставов (в градусах)
theta_yaw = 0.0       # Вращение базы
theta_shoulder = 30.0 # Плечо
theta_elbow = -45.0   # Локоть

JOINT_LIMIT = 89.0    # Предельный наклон плеча/локтя, чтобы рука не уходила под пол
ROT_SPEED = 60        # Скорость вращения суставов, град/сек


def update():
    global theta_yaw, theta_shoulder, theta_elbow

    # Плавное ручное управление с клавиатуры для проверки:
    # A / D -> вращение базы
    # W / S -> наклон плеча
    # Q / E -> сгиб локтя
    if held_keys['d']: theta_yaw += ROT_SPEED * time.dt
    if held_keys['a']: theta_yaw -= ROT_SPEED * time.dt
    if held_keys['w']: theta_shoulder += ROT_SPEED * time.dt
    if held_keys['s']: theta_shoulder -= ROT_SPEED * time.dt
    if held_keys['e']: theta_elbow += ROT_SPEED * time.dt
    if held_keys['q']: theta_elbow -= ROT_SPEED * time.dt

    # Ограничиваем углы физическими пределами
    theta_shoulder = max(-JOINT_LIMIT, min(JOINT_LIMIT, theta_shoulder))
    theta_elbow = max(-JOINT_LIMIT, min(JOINT_LIMIT, theta_elbow))

    # Применяем углы поворота
    base_yaw.rotation_y = theta_yaw
    shoulder_pivot.rotation_x = theta_shoulder
    elbow_pivot.rotation_x = theta_elbow

# Текстовая подсказка на экране
Text("Управление камерой: Зажми ПКМ для вращения, колесико для зума", position=(-0.85, 0.45), scale=1.1)
Text("Тест суставов: [A/D] - База, [W/S] - Плечо, [Q/E] - Локоть", position=(-0.85, 0.40), scale=1.1, color=color.yellow)

app.run()
