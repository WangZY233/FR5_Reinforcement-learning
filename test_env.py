from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from Fr5 import Fr5
from Fr5_task import Fr5_task
import time


class MyRobotTaskEnv(RobotTaskEnv):
    """My robot-task environment."""

    def __init__(self, render_mode):
        sim = PyBullet(render_mode=render_mode)
        robot = Fr5(sim)
        task = Fr5_task(sim)
        super().__init__(robot, task)
    
env = MyRobotTaskEnv(render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        time.sleep(1)
    