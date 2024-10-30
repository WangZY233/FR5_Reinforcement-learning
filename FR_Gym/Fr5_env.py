'''
 @Author: Prince Wang 
 @Date: 2024-02-22 
 @Last Modified by:   Prince Wang 
 @Last Modified time: 2023-10-24 23:04:04 
'''
import os

import pybullet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import time
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from loguru import logger
import random
from reward import grasp_reward


class FR5_Env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, gui=False):
        super(FR5_Env).__init__()
        self.step_num = 0
        self.Con_cube = None
        # self.last_success = False

        # 设置最小的关节变化量
        low_action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        high_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        low = np.zeros((1, 14), dtype=np.float32)
        high = np.ones((1, 14), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 初始化pybullet环境
        if gui == False:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
        else:
            self.p = bullet_client.BulletClient(connection_mode=p.GUI)
        # self.p.setTimeStep(1/240)
        # print(self.p)
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 初始化环境
        self.init_env()

    def init_env(self):
        '''
            仿真环境初始化
        '''
        # boxId = self.p.loadURDF("plane.urdf")

        # 创建机械臂
        self.fr5 = self.p.loadURDF(
            "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning/fr5_description/urdf/fr5v6.urdf",
            useFixedBase=True, basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
            flags=p.URDF_USE_SELF_COLLISION
        )

        # 创建桌子
        self.table = p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.63],
                                baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))

        # 创建目标
        collisionTargetId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                        radius=0.02, height=0.05)
        self.target = self.p.createMultiBody(baseMass=0,  # 质量
                                             baseCollisionShapeIndex=collisionTargetId,
                                             basePosition=[0.5, 0.5, 2])
        p.changeDynamics(self.target, -1, lateralFriction=10.0, spinningFriction=1, rollingFriction=1)
        # self.target = self.p.loadURDF(
        #     "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning/fr5_description/urdf/box.urdf",
        #     basePosition=[0.5, 0.5, 2])

        # 创建目标杯子的台子
        collisionTargetId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                        radius=0.03, height=0.3)
        self.targettable = self.p.createMultiBody(baseMass=0,  # 质量
                                                  baseCollisionShapeIndex=collisionTargetId,
                                                  basePosition=[0.5, 0.5, 2])
        # # 创建障碍物
        # obstacleId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
        #                                          radius=0.02, height=0.4)
        # self.obstacle = self.p.createMultiBody(baseMass=0,  # 质量
        #                                        baseCollisionShapeIndex=obstacleId,
        #                                        basePosition=[0.5, 0.5, 2])
        #

    def step(self, action):
        '''step'''
        info = {}
        # Execute one time step within the environment
        # 初始化关节角度列表
        joint_angles = []
        # 初始化夹爪位置
        gripper = [0.0, 0.0]

        # 获取每个关节的状态
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angle = joint_info[0]  # 第一个元素是当前关节角度
            joint_angles.append(joint_angle)

        # # 获取夹爪位置
        # last_position = self.get_gripper_position()
        # Euler = [last_position[3], last_position[4], last_position[5]+action[3]]
        # orn = p.getQuaternionFromEuler(Euler)
        # pos = [last_position[0] + action[0], last_position[1] + action[1], last_position[2] + action[2]]
        #
        # # pos = [-0.021248765289783478, 0.18191272020339966, 0.30001968145370483]
        # joint_angles = p.calculateInverseKinematics(self.fr5, 6, pos, orn)

        # 执行action
        Fr5_joint_angles = np.array(joint_angles[:6]) + (np.array(action[0:6]) / 180 * np.pi)

        if action[6] > 0:
            gripper = np.array([0.0, 0.0])
        else:
            gripper = np.array([0.0, 0.0])

        anglenow = np.hstack([Fr5_joint_angles, gripper])
        anglenow[5] = 0
        p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                    targetPositions=anglenow)

        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1./240.)

        self.reward, info = grasp_reward(self)

        # observation计算
        self.get_observation()

        self.step_num += 1

        return self.observation, self.reward, self.terminated, self.truncated, info

    def moveTarget(self):
        # 移动目标位置
        delta_x = np.random.uniform(-0.1, 0.1, 1)
        delta_y = np.random.uniform(-0.1, 0.1, 1)
        self.now_target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])
        self.target_position = self.now_target_position + np.array([delta_x[0], delta_y[0], 0])
        p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])
        self.targettable_position = self.target_position + np.array([0, 0, -0.16])
        p.resetBasePositionAndOrientation(self.targettable, self.targettable_position, [0, 0, 0, 1])

    def reset(self, seed=None, options=None):
        '''重置环境参数'''
        self.step_num = 0
        self.reward = 0
        self.terminated = False
        self.success = False
        # 重新设置机械臂的位置
        neutral_angle = [30, -137, 128, 9,
                         30, 0, 0, 0]
        neutral_angle = [x * math.pi / 180 for x in neutral_angle]
        p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                    targetPositions=neutral_angle)

        # # 重新设置目标位置
        # self.goalx = np.random.uniform(-0.2, 0.2, 1)
        # self.goaly = np.random.uniform(0.6, 0.8, 1)
        # self.goalz = np.random.uniform(0.1, 0.3, 1)
        # self.target_position = [self.goalx[0], self.goaly[0], self.goalz[0]]
        self.target_position = [0., 0.6, 0.2]
        # self.targettable_position = [self.goalx[0], self.goaly[0], self.goalz[0] - 0.175]
        self.targettable_position = [0, 0.6, 0.025]
        self.p.resetBasePositionAndOrientation(self.targettable, self.targettable_position, [0, 0, 0, 1])
        self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])
        self.ori_target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])

        # # 重新设置障碍物位置
        # self.obstacle_deltax = np.random.uniform(-0.1, 0.1, 1)
        # self.obstacle_deltay = np.random.uniform(-0.1, -0.2, 1)
        # self.obstacle_position = [self.obstacle_deltax[0] + self.goalx[0], self.obstacle_deltay[0] + self.goaly[0],
        #                           self.goalz[0]+0.05]
        # self.p.resetBasePositionAndOrientation(self.obstacle, self.obstacle_position, [0, 0, 0, 1])

        for i in range(100):
            self.p.stepSimulation()
            # time.sleep(10./240.)

        self.get_observation()

        infos = {}
        infos['is_success'] = False
        infos['reward'] = 0
        infos['step_num'] = 0
        # print("observation", self.observation)
        return self.observation, infos

    def get_gripper_position(self):
        '''获取夹爪中心位置和朝向'''
        Gripper_pos = p.getLinkState(self.fr5, 6)[4]
        # relative_position = np.array([0, 0, 0.15])

        rotation = pybullet.getEulerFromQuaternion(p.getLinkState(self.fr5, 6)[5])
        # rotated_relative_position = rotation.apply(relative_position)
        # gripper_centre_pos = [Gripper_posx, Gripper_posy, Gripper_posz] + rotated_relative_position
        # gripper_centre_pos = p.getJointState(self.fr5, 6)

        gripper_centre_pos = Gripper_pos[:3] + rotation

        return gripper_centre_pos

    def get_observation(self, add_noise=False):
        """计算observation"""
        Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
        Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
        Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
        relative_position = np.array([0, 0, 0.22])

        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = [Gripper_posx, Gripper_posy, Gripper_posz] + rotated_relative_position

        joint_angles = [0, 0, 0, 0, 0, 0]
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angles[i - 1] = joint_info[0] * 180 / np.pi  # 第一个元素是当前关节角度
            if add_noise == True:
                joint_angles[i - 1] = self.add_noise(joint_angles[i - 1], range=0, gaussian=True)

        gripper_angle = []
        for i in [8, 9]:
            joint_info = p.getJointState(self.fr5, i)
            gripper_angle.append(joint_info[0])

        # 计算夹爪的朝向
        gripper_orientation = p.getLinkState(self.fr5, 7)[1]
        gripper_orientation = R.from_quat(gripper_orientation)
        gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)

        # 计算obs
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        obs_gripper_angles = np.array(gripper_angle, dtype=np.float32)
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0] + 0.922) / 1.844,
                                           (gripper_centre_pos[1] + 0.922) / 1.844,
                                           (gripper_centre_pos[2] + 0.5) / 1], dtype=np.float32)

        obs_gripper_orientation = (np.array([gripper_orientation[0], gripper_orientation[1], gripper_orientation[2]],
                                            dtype=np.float32) + 180) / 360

        self.target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])

        obs_target_position = np.array([(self.target_position[0] + 0.2) / 0.4,
                                        (self.target_position[1] - 0.6) / 0.2,
                                        (self.target_position[2] - 0.1) / 0.2], dtype=np.float32)

        # self.obstacle_position = np.array(p.getBasePositionAndOrientation(self.obstacle)[0])
        #
        # obs_obstacle_position = np.array([(self.obstacle_position[0] + 0.3) / 0.3,
        #                                 (self.obstacle_position[1] - 0.55) / 0.3,], dtype=np.float32)

        self.observation = np.hstack(
            (obs_gripper_centre_pos, obs_joint_angles, obs_gripper_angles, obs_target_position)).flatten()

        self.observation = self.observation.flatten()
        self.observation = self.observation.reshape(1, 14)

    def get_observation_(self, add_noise=False):
        """计算observation"""
        gripper_centre_pos = self.get_gripper_position()
        obs_gripper_centre_pos = np.array(gripper_centre_pos, dtype=np.float32)

        joint_angles = [0, 0, 0, 0, 0, 0]
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angles[i - 1] = joint_info[0] * 180 / np.pi  # 第一个元素是当前关节角度
            if add_noise == True:
                joint_angles[i - 1] = self.add_noise(joint_angles[i - 1], range=0, gaussian=True)
        joint_angles = np.array(joint_angles, dtype=np.float32)
        self.target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])

        obs_target_position = np.array([self.target_position[0],
                                        self.target_position[1],
                                        self.target_position[2]], dtype=np.float32)
        self.observation = np.hstack(
            (obs_gripper_centre_pos, obs_target_position, joint_angles)).flatten()

        self.observation = self.observation.flatten()
        self.observation = self.observation.reshape(1, 15)

    def render(self):
        '''设置观察角度'''
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0, cameraYaw=90, cameraPitch=-7.6, cameraTargetPosition=[0.39, 0.45, 0.42])

    def close(self):
        self.p.disconnect()

    def add_noise(self, angle, range, gaussian=False):
        '''添加噪声'''
        if gaussian:
            angle += np.clip(np.random.normal(0, 1) * range, -1, 1)
        else:
            angle += random.uniform(-5, 5)
        return angle


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    Env = FR5_Env(gui=True)
    Env.reset()
    check_env(Env, warn=True)
    # for i in range(100):
    #         p.stepSimulation()
    #         time.sleep(1./240.)
    Env.render()
    print("test going")
    time.sleep(10)
    # observation, reward, terminated, truncated, info = Env.step([0,0,0,0,0,20])
    # print(reward)
    time.sleep(100)
