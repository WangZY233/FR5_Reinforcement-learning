from panda_gym.pybullet import PyBullet
from Fr5_task import MyTask
import time

sim = PyBullet(render_mode="human")
task = MyTask(sim)

for _ in range(10):
    task.reset()
    print(task.get_obs())
    print(task.get_achieved_goal())
    print(task.is_success(task.get_achieved_goal(), task.get_goal()))
    print(task.compute_reward(task.get_achieved_goal(), task.get_goal()))
    time.sleep(1)