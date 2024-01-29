import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance
from Fr5 import Fr5


class Fr5_task(Task):
    def __init__(self, sim):
        super().__init__(sim)
        # create an cube
        self.sim.create_box(body_name="target", half_extents=np.array([0.02, 0.02, 0.02]), mass=0, position=np.array([0.2, -0.2, 0.0]), rgba_color = np.array([255,255,255]))
        self.robot = Fr5

    def reset(self):
        # randomly sample a goal position
        self.goalx = np.random.uniform(-0.3, 0.3, 1)
        self.goaly = np.random.uniform(0.2, 0.5, 1)
        self.goal = np.array([self.goalx[0], self.goaly[0], 0.0])
        # print(self.goaly)
        # reset the position of the object
        self.sim.set_base_pose("target", position=np.array([self.goalx[0], self.goaly[0], 0.0]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))

    def get_obs(self):
        # the observation is the position of the object
        observation = self.sim.get_base_position("target")
        return observation

    def get_achieved_goal(self):
        # the achieved goal is the current position of the object
        # achieved_goal = self.sim.get_base_position("target")
        achieved_goal = self.sim.get_link_position("Fr5",6)
        # print(achieved_goal)
        return achieved_goal

    def is_success(self, achieved_goal, desired_goal, info={}):  # info is here for consistancy
        # compute the distance between the goal position and the current object position
        d = distance(achieved_goal, desired_goal)
        # return True if the distance is < 1.0, and False otherwise
        return np.array(d < 1.0, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info={}):  # info is here for consistancy
        # for this example, reward = 1.0 if the task is successfull, 0.0 otherwise
        return self.is_success(achieved_goal, desired_goal, info).astype(np.float32)