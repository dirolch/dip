import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
from ursina import *

# Настройки среды
L1, L2 = 2.0, 1.6
STEP = 3.5  # градусы

# Создаем материал для металлического вида
def create_material():
    return Material(diffuse_color=color.gray, specular=0.5, roughness=0.3)

# Создаем текстуру для базы
def create_base_material():
    return Material(color=color.gray, texture='metal')
# Иерархия манипулятора
base_stand = Entity(model='cylinder', color=color.gray, scale=(1.2, 0.4, 0.2), y=0.2, material=create_base_material)
base_yaw = Entity(model='cube', color=color.azure, scale=(0.8, 0.6, 0.8), y=0.5, material=create_base_material)

shoulder_pivot = Entity(parent=base_yaw, y=0.3)
shoulder_link = Entity(model='cone', color=color.cyan, scale=(0.3, L1, 0.3), y=L1/2, material=create_base_material)
elbow_pivot = Entity(parent=shoulder_pivot, y=L1)
elbow_link = Entity(model='cylinder', color=color.orange, scale=(0.25, L2, 0.25), y=L2/2, material=create_base_material)

wrist = Entity(parent=elbow_pivot, y=L2)
end_effector = Entity(model='sphere', color=color.yellow, scale=0.3, material=create_base_material)

target = Entity(model='sphere', color=color.red, scale=0.35, material=create_base_material)

# Использование анимации для плавного движения

app = Ursina()
EditorCamera()

class ArmSimulation(Entity):
    def __init__(self):
        super().__init__()
        self.create_base()
        self.create_joint()
        self.create_links()
        self.create_base_stand()
        self.create_target()
        self.create_info_text()

    def create_base(self):
        self.base = Entity(model='cylinder', collider=True, color=color.gray, scale=(1.2, 0.4, 0.2), y=0.2)

    def create_joint(self):
        self.joint = Entity(model='cube', position=(0, 0, 0), color=color.blue, scale_x=0.5, scale_y=0.5, scale_z=0.5)

    def create_links(self):
        self.link1 = Entity(model='cube', color=color.red, scale=(0.2, 0.2, 1))
        self.link2 = Entity(model='cube', color=color.green, scale=(0.2, 0.2, 1))

    def create_base_stand(self):
        self.stand = Entity(model='cylinder', color=color.gray, scale=(1.2, 0.4, 0.2), y=0.2)

    def create_target(self):
        self.target = Entity(model='sphere', color=color.yellow, scale=0.1)

    def create_info_text(self):
        self.info = Text(text=f"Успешно загружен модель: 'cylinder'", position=(-1, 0, 0))

arm = ArmSimulation()
arm.run()

