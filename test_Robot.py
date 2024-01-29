from panda_gym.pybullet import PyBullet
import gymnasium as gym
import panda_gym
from Fr5 import Fr5
import numpy as np
import time

env = gym.make("PandaReach-v3", render_mode="rgb_array", renderer="OpenGL")
env.reset()
image = env.render()  # RGB rendering of shape (480, 720, 3)
env.close()
sim = PyBullet(render_mode="human")
robot = Fr5(sim)
Fr5.inverse_kinematics
# env = gym.make(
#     "PandaSlide-v3",
#     render_mode="rgb_array",
#     renderer="OpenGL",
#     render_width=480,
#     render_height=480,
#     render_target_position=[0.2, 0, 0],
#     render_distance=1.0,
#     render_yaw=90,
#     render_pitch=-70,
#     render_roll=0,
# # )
# env.reset()
# time.sleep(10)
# image = env.render()  # RGB rendering of shape (480, 480, 3)
# env.close()

# for _ in range(50):
#     robot.set_action(np.array([1.0,0,0,0,0,0,0]))
#     sim.step()
#     time.sleep(1)