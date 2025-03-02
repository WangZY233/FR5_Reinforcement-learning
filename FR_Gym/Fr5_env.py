'''
 @Author: Prince Wang 
 @Date: 2024-02-22 
 @Last Modified by:   Prince Wang 
 @Last Modified time: 2023-10-24 23:04:04 
'''
import os
import pybullet
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
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
from stable_baselines3 import PPO


class FR5_Env(gym.Env):
    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, gui=False, use_guide_model=True, test=False, stage_skip_rate=0,
                 guide_rate=1.0, online=False ,info=""):
        super(FR5_Env).__init__()
        self.use_stage_skip = False  #是否启用阶段跳过以实现数据采集
        self.stage_skip_rate = stage_skip_rate  #阶段跳过的概率
        self.test = test
        self.step_num = 0
        self.Con_cube = None
        self.stage = 0
        self.use_guide_model = use_guide_model  #是否以一定概率使用指导模型动作
        self.guide_rate = guide_rate if test is False else 0  #指导模型动作的概率
        self.info = info
        self.online = online
        if self.online is False:
            self.model_0 = PPO.load(
                "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/pick_model")
            self.model_1 = PPO.load(
                "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/place_model")
            self.model_2 = PPO.load(
                "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/button_model")
            self.model_3 = PPO.load(
                "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/catch_model")
            self.model_4 = PPO.load(
                "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/trans_model")
        else:
            self.model_0 = PPO.load(
                "/root/FR5/models/pick_model")
            self.model_1 = PPO.load(
                "/root/FR5/models/place_model")
            self.model_2 = PPO.load(
                "/root/FR5/models/button_model")
            self.model_3 = PPO.load(
                "/root/FR5/models/catch_model")
            self.model_4 = PPO.load(
                "/root/FR5/models/trans_model")

        # self.last_success = False
        print('模型载入完毕')

        self.grasp_zero = [0, 0]
        self.grasp_zero_sym = [0.075, 0.075]
        self.grasp_effort_sym = [0.065, 0.065]
        self.grasp_effort_ori = [0.003, 0.003]
        # 设置最小的关节变化量
        low_action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        high_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        low = np.zeros((1, 16), dtype=np.float32)
        high = np.ones((1, 16), dtype=np.float32)
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
        # self.first_model = PPO.load(
        #     "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_0/FR_Gym/FR5_Reinforcement-learning/models/PPO/1107-121135/pick_model.zip")
        # 创建机械臂
        if self.online is False:
            self.fr5 = self.p.loadURDF(
                "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/fr5_description/urdf/fr5v6.urdf",
                useFixedBase=True, basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
                flags=p.URDF_USE_SELF_COLLISION
            )
        else:
            self.fr5 = self.p.loadURDF(
                "/root/FR5/fr5_description/urdf/fr5v6.urdf",
                useFixedBase=True, basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
                flags=p.URDF_USE_SELF_COLLISION
            )

        # 创建桌子
        self.table = p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.63],
                                baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))

        # 创建按钮
        self.button_length = 0.01
        self.button_height = 0.3
        buttonId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                               radius=0.015, height=self.button_length)

        self.button = self.p.createMultiBody(baseMass=0,  # 质量
                                             baseCollisionShapeIndex=buttonId,
                                             basePosition=[0.5, 0.5, 2])

        # 创建目标
        self.cup_height = 0.1
        collisionTargetId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                        radius=0.025, height=self.cup_height)

        self.target = self.p.createMultiBody(baseMass=0.2,  # 质量
                                             baseCollisionShapeIndex=collisionTargetId,
                                             basePosition=[0.5, 0.5, 2])

        self.grasp_effort = [0.0080, 0.0080]
        self.grasp_zero = [0, 0]
        self.grasp_sym = [0.85, 0.85]
        self.grasp_tauch = [0.15, 0.15]
        p.changeDynamics(self.target, -1, lateralFriction=0.1, spinningFriction=0.1, rollingFriction=0.1)

        # 创建目标杯子的台子
        self.targettable_height = 0.04
        tableId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                              radius=0.04, height=self.targettable_height)
        self.targettable = self.p.createMultiBody(baseMass=0,  # 质量
                                                  baseCollisionShapeIndex=tableId,
                                                  basePosition=[0.5, 0.5, 2])
        p.changeDynamics(self.targettable, -1, lateralFriction=0.1, spinningFriction=0, rollingFriction=0)

        # # 创建障碍物
        self.obstacle_wide = 0.05
        obstacleId = self.p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                 halfExtents=[self.obstacle_wide * 2, self.obstacle_wide, 0.1])

        self.obstacle = self.p.createMultiBody(baseMass=0,  # 质量
                                               baseCollisionShapeIndex=obstacleId,
                                               basePosition=[0.5, 0.5, 2])

        # 创建目标台子
        self.targettable_2_height = 0.04
        targettable_2Id = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                      radius=0.04, height=self.targettable_2_height)
        self.targettable_2 = self.p.createMultiBody(baseMass=0,  # 质量
                                                    baseCollisionShapeIndex=targettable_2Id,
                                                    basePosition=[0.5, 0.5, 2])

    def step(self, action):
        if self.stage == 0:
            return self.step_0(action)
        elif self.stage == 1:
            return self.step_1(action)
        elif self.stage == 2:
            return self.step_2(action)
        elif self.stage == 3:
            return self.step_3(action)
        elif self.stage == 4:
            return self.step_4(action)

    def step_0(self, action):
        '''step'''
        info = {}
        # Execute one time step within the environment
        guide_action, _ = self.model_0.predict(observation=self.guide_observation, deterministic=True)

        # 比较两个动作的差异
        diff = np.sum(abs(guide_action[:6] - action[:6]))

        # 初始化关节角度列表
        joint_angles = []

        # 获取每个关节的状态
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angle = joint_info[0]  # 第一个元素是当前关节角度
            joint_angles.append(joint_angle)

        # 执行action
        if self.use_guide_model and np.random.uniform(0, 1) < self.guide_rate:
            action = guide_action
        Fr5_joint_angles = np.array(joint_angles[:6]) + (np.array(action[0:6]) / 180 * np.pi)

        gripper = np.array([0, 0])

        anglenow = np.hstack([Fr5_joint_angles, gripper])
        p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                    targetPositions=anglenow)

        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1./240.)

        self.reward, info = grasp_reward(self, diff)

        # observation计算
        self.get_observation_0()
        self.get_observation()

        self.step_num += 1
        return self.observation, self.reward, self.terminated, self.truncated, info

    def step_1(self, action):
        '''step'''
        info = {}
        # Execute one time step within the environment
        guide_action, _ = self.model_1.predict(observation=self.guide_observation, deterministic=True)

        # 比较两个动作的差异
        diff = np.sum(abs(guide_action[:6] - action[:6]))

        # 初始化关节角度列表
        joint_angles = []

        # 获取每个关节的状态
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angle = joint_info[0]  # 第一个元素是当前关节角度
            joint_angles.append(joint_angle)

        # 执行action
        if self.use_guide_model and np.random.uniform(0, 1) < self.guide_rate:
            action = guide_action
        Fr5_joint_angles = np.array(joint_angles[:6]) + (np.array(action[0:6]) / 180 * np.pi)
        gripper = np.array(self.grasp_effort)

        anglenow = np.hstack([Fr5_joint_angles, gripper])
        p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                    targetPositions=anglenow)

        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1./240.)

        self.reward, info = grasp_reward(self, diff)

        # observation计算
        self.get_observation_1()
        self.get_observation()

        self.step_num += 1

        return self.observation, self.reward, self.terminated, self.truncated, info

    def step_2(self, action):
        '''step'''
        info = {}
        # Execute one time step within the environment
        guide_action, _ = self.model_2.predict(observation=self.guide_observation, deterministic=True)

        # 比较两个动作的差异
        diff = np.sum(abs(guide_action[:6] - action[:6]))

        # 初始化关节角度列表
        joint_angles = []
        # 初始化夹爪位置
        gripper = [0.0, 0.0]

        # 获取每个关节的状态
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angle = joint_info[0]  # 第一个元素是当前关节角度
            joint_angles.append(joint_angle)

        # 执行action
        if self.use_guide_model and np.random.uniform(0, 1) < self.guide_rate:
            action = guide_action

        Fr5_joint_angles = np.array(joint_angles[:6]) + (np.array(action[0:6]) / 180 * np.pi)

        # if self.step_num > 20:
        #     gripper = np.array(self.grasp_tauch)
        # else:
        #     gripper = np.array([0, 0])
        gripper = np.array([0, 0])
        anglenow = np.hstack([Fr5_joint_angles, gripper])
        anglenow[5] = 0
        p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                    targetPositions=anglenow)

        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1./240.)

        self.reward, info = grasp_reward(self, diff)

        # observation计算
        self.get_observation_2()
        self.get_observation()

        self.step_num += 1

        return self.observation, self.reward, self.terminated, self.truncated, info

    def step_3(self, action):
        '''step'''
        info = {}
        # Execute one time step within the environment
        guide_action, _ = self.model_3.predict(observation=self.guide_observation, deterministic=True)

        # 比较两个动作的差异
        diff = np.sum(abs(guide_action[:6] - action[:6]))

        # 初始化关节角度列表
        joint_angles = []
        # 初始化夹爪位置
        gripper = [0.0, 0.0]

        # 获取每个关节的状态
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angle = joint_info[0]  # 第一个元素是当前关节角度
            joint_angles.append(joint_angle)

        # 执行action
        if self.use_guide_model and np.random.uniform(0, 1) < self.guide_rate:
            action = guide_action
        Fr5_joint_angles = np.array(joint_angles[:6]) + (np.array(action[0:6]) / 180 * np.pi)

        gripper = np.array([0, 0])

        anglenow = np.hstack([Fr5_joint_angles, gripper])
        anglenow[5] = 0
        p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                    targetPositions=anglenow)

        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1./240.)

        self.reward, info = grasp_reward(self, diff)

        # observation计算
        self.get_observation_3()
        self.get_observation()
        self.step_num += 1

        return self.observation, self.reward, self.terminated, self.truncated, info

    def step_4(self, action):
        '''step'''
        info = {}
        # Execute one time step within the environment
        guide_action, _ = self.model_4.predict(observation=self.guide_observation, deterministic=True)

        # 比较两个动作的差异
        diff = np.sum(abs(guide_action[:6] - action[:6]))

        # 初始化关节角度列表
        joint_angles = []

        # 获取每个关节的状态
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angle = joint_info[0]  # 第一个元素是当前关节角度
            joint_angles.append(joint_angle)

        # 执行action
        if self.use_guide_model and np.random.uniform(0, 1) < self.guide_rate:
            action = guide_action
        Fr5_joint_angles = np.array(joint_angles[:6]) + (np.array(action[0:6]) / 180 * np.pi)

        gripper = np.array(self.grasp_effort)

        anglenow = np.hstack([Fr5_joint_angles, gripper])
        anglenow[5] = 0
        p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                    targetPositions=anglenow)

        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1./240.)

        self.reward, info = grasp_reward(self, diff)

        # observation计算
        self.get_observation_4()
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

    def flashUR5(self):
        """夹爪干涉出现时，反置机械臂"""
        if self.grasp_zero[0] == 0:
            self.grasp_zero = self.grasp_zero_sym
            self.grasp_effort = self.grasp_effort_sym
        elif self.grasp_zero[0] == self.grasp_zero_sym[0]:
            self.grasp_zero = [0, 0]
            self.grasp_effort = self.grasp_effort_ori

    def reset(self, seed=None, options=None):
        if self.use_stage_skip == True and np.random.uniform(0, 1) < self.stage_skip_rate:  #(1 - 1/(5-self.stage))
            return self.stage_skip()
        else:
            if self.stage == 0:
                return self.reset_0(seed, options)
            elif self.stage == 1:
                return self.reset_1(seed, options)
            elif self.stage == 2:
                return self.reset_2(seed, options)
            elif self.stage == 3:
                return self.reset_3(seed, options)
            elif self.stage == 4:
                return self.reset_4(seed, options)

    def reset_0(self, seed=None, options=None):
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
        for _ in range(50):
            self.p.stepSimulation()
        '''干涉检测'''
        error_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.fr5)
        for contact_point in error_contact_points:
            link_index = contact_point[3]
            if link_index == 7 or link_index == 8:
                logger.info("夹爪干涉出现！")
                self.flashUR5()
                for _ in range(10):
                    self.p.stepSimulation()
                break
        # # 重新设置目标位置
        # self.goalx = np.random.uniform(0.1, 0.35, 1)[0]
        # self.goaly = np.random.uniform(0.45, 0.6, 1)[0]
        # self.goalz = np.random.uniform(0.06, 0.1, 1)[0]
        # 减小随机性
        self.goalx = np.random.uniform(0.2, 0.3, 1)[0]
        self.goaly = np.random.uniform(0.5, 0.6, 1)[0]
        self.goalz = np.random.uniform(0.06, 0.08, 1)[0]
        self.target_position = [self.goalx, self.goaly, self.goalz]
        # self.target_position = [0., 0.6, 0.2]
        self.targettable_2_position = [self.goalx, self.goaly,
                                       self.goalz - self.cup_height / 2 - self.targettable_height / 2]

        self.p.resetBasePositionAndOrientation(self.targettable_2, self.targettable_2_position, [0, 0, 0, 1])
        self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])
        self.ori_target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])

        for _ in range(100):
            self.p.stepSimulation()
            # time.sleep(10./240.)

        self.get_observation()
        self.get_observation_0()
        infos = {}
        infos['is_success'] = False
        infos['reward'] = 0
        infos['step_num'] = 0
        # print("observation", self.observation)
        return self.observation, infos

    def reset_1(self, seed=None, options=None):
        """重置环境参数"""
        self.step_num = 0
        self.reward = 0
        self.terminated = False
        self.success = False

        # 重新设置目标咖啡机位置
        # self.goalx = np.random.uniform(-0.2, 0.2, 1)[0]
        # self.goaly = np.random.uniform(0.8, 0.9, 1)[0]
        # self.goalz = self.targettable_height + self.cup_height / 2
        # 减小随机性
        self.goalx = np.random.uniform(0.1, 0.2, 1)[0]
        self.goaly = np.random.uniform(0.8, 0.9, 1)[0]
        self.goalz = self.targettable_height + self.cup_height / 2
        self.base_position = [self.goalx, self.goaly, self.button_height]
        self.p.resetBasePositionAndOrientation(self.obstacle, self.base_position, [0, 0, 0, 1])
        Euler = [math.pi / 2, 0, 0]
        key_orn = p.getQuaternionFromEuler(Euler)

        # self.target_position = [0.5, 0.5, 1]
        self.button_position = [self.goalx, self.goaly - self.obstacle_wide, self.button_height]
        self.p.resetBasePositionAndOrientation(self.button, self.button_position, key_orn)

        self.targettable_position = [self.goalx, self.goaly, self.targettable_height / 2]
        self.p.resetBasePositionAndOrientation(self.targettable, self.targettable_position, [0, 0, 0, 1])

        # 设置初始杯子位置
        # gripper_centre_pos = self.get_gripper_position()
        # self.base_position = [gripper_centre_pos[0], gripper_centre_pos[1], self.targettable_2_height / 2]
        # self.p.resetBasePositionAndOrientation(self.targettable_2, self.base_position, [0, 0, 0, 1])
        # self.target_position = [gripper_centre_pos[0], gripper_centre_pos[1],
        #                         self.targettable_2_height + self.cup_height / 2]
        # self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])

        self.ori_target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])
        # 合上夹爪
        p.setJointMotorControlArray(self.fr5, [8, 9], p.POSITION_CONTROL,
                                    targetPositions=self.grasp_effort)
        for i in range(100):
            self.p.stepSimulation()
            # time.sleep(1. / 240.)
        self.get_observation_1()
        self.get_observation()
        infos = {}
        infos['is_success'] = False
        infos['reward'] = 0
        infos['step_num'] = 0
        # print("observation", self.observation)
        return self.observation, infos

    def reset_2(self, seed=None, options=None):
        '''重置环境参数'''
        self.step_num = 0
        self.reward = 0
        self.terminated = False
        self.success = False
        self.goalz = self.button_height
        self.goaly = self.goaly - self.obstacle_wide - self.button_length / 2
        for i in range(100):
            self.p.stepSimulation()
            # time.sleep(1. / 240.)

        self.get_observation_2()
        self.get_observation()
        infos = {}
        infos['is_success'] = False
        infos['reward'] = 0
        infos['step_num'] = 0
        # print("observation", self.observation)
        return self.observation, infos

    def reset_3(self, seed=None, options=None):
        '''重置环境参数'''
        self.step_num = 0
        self.reward = 0
        self.terminated = False
        self.success = False
        self.goaly = self.goaly + self.obstacle_wide + self.button_length / 2
        self.goalz = self.cup_height / 2 + self.targettable_2_height
        self.get_observation_3()
        self.get_observation()
        # 松开夹爪
        p.setJointMotorControlArray(self.fr5, [8, 9], p.POSITION_CONTROL,
                                    targetPositions=[0, 0])
        for i in range(10):
            self.p.stepSimulation()
            # time.sleep(1. / 240.)

        infos = {}
        infos['is_success'] = False
        infos['reward'] = 0
        infos['step_num'] = 0
        # print("observation", self.observation)
        return self.observation, infos

    def reset_4(self, seed=None, options=None):
        '''重置环境参数'''
        self.step_num = 0
        self.reward = 0
        self.terminated = False
        self.success = False
        self.goalx = np.random.uniform(0.25, 0.35, 1)[0]
        self.goaly = np.random.uniform(0.6, 0.7, 1)[0]

        self.goalz = self.targettable_2_height + self.cup_height / 2
        self.targettable_2_position = [self.goalx, self.goaly,
                                       self.goalz - self.cup_height / 2 - self.targettable_height / 2 - 0.01]
        self.p.resetBasePositionAndOrientation(self.targettable_2, self.targettable_2_position, [0, 0, 0, 1])

        self.get_observation_4()
        self.get_observation()
        # 和上夹爪
        p.setJointMotorControlArray(self.fr5, [8, 9], p.POSITION_CONTROL,
                                    targetPositions=self.grasp_effort)
        for i in range(10):
            self.p.stepSimulation()
            # time.sleep(1. / 240.)
        infos = {}
        infos['is_success'] = False
        infos['reward'] = 0
        infos['step_num'] = 0
        # print("observation", self.observation)
        return self.observation, infos

    def stage_skip(self):
        '''阶段跳过'''
        # 执行当前阶段对应指导动作，直到结束，然后进入下一个阶段
        if self.stage == 0:
            done = False
            state, _ = self.reset_0()
            while not done:
                action, _ = self.model_0.predict(observation=self.guide_observation, deterministic=True)
                state, reward, done, _, info = self.step_0(action=action)
            return self.reset()  # 由于最后未必执行成功，所以需要根据stage进行重置
        elif self.stage == 1:
            done = False
            state, _ = self.reset_1()
            while not done:
                action, _ = self.model_1.predict(observation=self.guide_observation, deterministic=True)
                state, reward, done, _, info = self.step_1(action=action)
            return self.reset()
        elif self.stage == 2:
            done = False
            state, _ = self.reset_2()
            while not done:
                action, _ = self.model_2.predict(observation=self.guide_observation, deterministic=True)
                state, reward, done, _, info = self.step_2(action=action)
            return self.reset()
        elif self.stage == 3:
            done = False
            state, _ = self.reset_3()
            while not done:
                action, _ = self.model_3.predict(observation=self.guide_observation, deterministic=True)
                state, reward, done, _, info = self.step_3(action=action)
            return self.reset()
        elif self.stage == 4:
            done = False
            state, _ = self.reset_4()
            while not done:
                action, _ = self.model_4.predict(observation=self.guide_observation, deterministic=True)
                state, reward, done, _, info = self.step_4(action=action)
            return self.reset()

    def get_gripper_position(self):
        '''获取夹爪中心位置和朝向'''
        Gripper_pos = p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.169])

        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = Gripper_pos + rotated_relative_position

        return gripper_centre_pos

    def get_observation(self, add_noise=False):
        """计算observation"""
        Gripper_pos = p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.169])

        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = Gripper_pos + rotated_relative_position

        joint_angles = [0, 0, 0, 0, 0, 0]
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angles[i - 1] = joint_info[0] * 180 / np.pi  # 第一个元素是当前关节角度
            if add_noise == True:
                joint_angles[i - 1] = self.add_noise(joint_angles[i - 1], range=0, gaussian=True)

        # 计算obs
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        # x,y,z坐标归一化
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0] + 0.5) / 1,
                                           (gripper_centre_pos[1] + 0.5) / 2,
                                           (gripper_centre_pos[2] + 0.5) / 1], dtype=np.float32)

        obs_target_position = np.array([(self.goalx + 0.4) / 0.8,
                                        self.goaly / 1,
                                        self.goalz / 0.5,
                                        self.stage / 5], dtype=np.float32)
        # 夹爪朝向
        obs_gripper_orientation = p.getLinkState(self.fr5, 7)[1]
        obs_gripper_orientation = R.from_quat(obs_gripper_orientation)
        obs_gripper_orientation = ((obs_gripper_orientation.as_euler('xyz', degrees=True) / 180) + 1) / 2

        self.observation = np.hstack(
            (obs_gripper_centre_pos, obs_joint_angles, obs_gripper_orientation, obs_target_position)).flatten()

        self.observation = self.observation.flatten()
        self.observation = self.observation.reshape(1, 16)

    def get_observation_0(self, add_noise=False):
        """计算observation"""
        Gripper_pos = p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.169])

        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = Gripper_pos + rotated_relative_position

        joint_angles = [0, 0, 0, 0, 0, 0]
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angles[i - 1] = joint_info[0] * 180 / np.pi  # 第一个元素是当前关节角度
            if add_noise == True:
                joint_angles[i - 1] = self.add_noise(joint_angles[i - 1], range=0, gaussian=True)

        # 计算obs
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        # x,y,z坐标归一化
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0] + 0.5) / 1,
                                           (gripper_centre_pos[1] + 0.5) / 2,
                                           (gripper_centre_pos[2] + 0.5) / 1], dtype=np.float32)

        obs_target_position = np.array([(self.goalx + 0.4) / 0.8,
                                        self.goaly / 1,
                                        self.goalz / 0.5], dtype=np.float32)
        # 夹爪朝向
        obs_gripper_orientation = p.getLinkState(self.fr5, 7)[1]
        obs_gripper_orientation = R.from_quat(obs_gripper_orientation)
        obs_gripper_orientation = ((obs_gripper_orientation.as_euler('xyz', degrees=True) / 180) + 1) / 2

        self.guide_observation = np.hstack(
            (obs_gripper_centre_pos, obs_joint_angles, obs_gripper_orientation, obs_target_position)).flatten()

        self.guide_observation = self.guide_observation.flatten()
        self.guide_observation = self.guide_observation.reshape(1, 15)

    def get_observation_1(self, add_noise=False):
        """计算observation"""
        Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
        Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
        Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
        relative_position = np.array([0, 0, 0.15])

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
        # print("夹爪旋转角度： ", pybullet.getEulerFromQuaternion(gripper_orientation))
        gripper_orientation = R.from_quat(gripper_orientation)
        gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)

        # 计算obs
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        obs_gripper_angles = np.array(gripper_angle, dtype=np.float32)
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0] + 0.922) / 1.844,
                                           (gripper_centre_pos[1] + 0.922) / 1.844,
                                           (gripper_centre_pos[2] + 0.5) / 1], dtype=np.float32)
        obs_target_position = np.array([(self.goalx + 0.5) / 1,
                                        (self.goaly - 0.5) / 0.1,
                                        (self.goalz - 0.05) / 0.1], dtype=np.float32)

        self.guide_observation = np.hstack(
            (obs_gripper_centre_pos, obs_joint_angles, obs_gripper_angles, obs_target_position)).flatten()

        self.guide_observation = self.guide_observation.flatten()
        self.guide_observation = self.guide_observation.reshape(1, 14)

    def get_observation_2(self, add_noise=False):
        """计算observation"""
        Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
        Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
        Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
        relative_position = np.array([0, 0, 0.15])

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
        # print("夹爪旋转角度： ", pybullet.getEulerFromQuaternion(gripper_orientation))
        gripper_orientation = R.from_quat(gripper_orientation)
        gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)

        # 计算obs
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        obs_gripper_angles = np.array(gripper_angle, dtype=np.float32)
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0] + 0.922) / 1.844,
                                           (gripper_centre_pos[1] + 0.922) / 1.844,
                                           (gripper_centre_pos[2] + 0.5) / 1], dtype=np.float32)
        obs_target_position = np.array([self.goalx + 0.4 / 0.8,
                                        (self.goaly - 0.7) / 0.25,
                                        (self.goalz - self.button_height + 0.05) / 0.1], dtype=np.float32)
        self.guide_observation = np.hstack(
            (obs_gripper_centre_pos, obs_joint_angles, obs_gripper_angles, obs_target_position)).flatten()

        self.guide_observation = self.guide_observation.flatten()
        self.guide_observation = self.guide_observation.reshape(1, 14)

    def get_observation_3(self, add_noise=False):
        """计算observation"""
        Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
        Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
        Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
        relative_position = np.array([0, 0, 0.15])

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
        # print("夹爪旋转角度： ", pybullet.getEulerFromQuaternion(gripper_orientation))
        gripper_orientation = R.from_quat(gripper_orientation)
        gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)

        # 计算obs
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        obs_gripper_angles = np.array(gripper_angle, dtype=np.float32)
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0] + 0.922) / 1.844,
                                           (gripper_centre_pos[1] + 0.922) / 1.844,
                                           (gripper_centre_pos[2] + 0.5) / 1], dtype=np.float32)

        obs_target_position = np.array([self.goalx + 0.4 / 0.8,
                                        (self.goaly - 0.7) / 0.25,
                                        (self.goalz + 0.05) / 0.1], dtype=np.float32)

        self.guide_observation = np.hstack(
            (obs_gripper_centre_pos, obs_joint_angles, obs_gripper_angles, obs_target_position)).flatten()

        self.guide_observation = self.guide_observation.flatten()
        self.guide_observation = self.guide_observation.reshape(1, 14)

    def get_observation_4(self, add_noise=False):
        """计算observation"""
        Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
        Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
        Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
        relative_position = np.array([0, 0, 0.15])

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
        # print("夹爪旋转角度： ", pybullet.getEulerFromQuaternion(gripper_orientation))
        gripper_orientation = R.from_quat(gripper_orientation)
        gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)

        # 计算obs
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        obs_gripper_angles = np.array(gripper_angle, dtype=np.float32)
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0] + 0.922) / 1.844,
                                           (gripper_centre_pos[1] + 0.922) / 1.844,
                                           (gripper_centre_pos[2] + 0.5) / 1], dtype=np.float32)

        obs_target_position = np.array([self.goalx + 0.4 / 0.8,
                                        (self.goaly - 0.3) / 0.4,
                                        (self.goalz) / 0.1], dtype=np.float32)
        self.guide_observation = np.hstack(
            (obs_gripper_centre_pos, obs_joint_angles, obs_gripper_angles, obs_target_position)).flatten()

        self.guide_observation = self.guide_observation.flatten()
        self.guide_observation = self.guide_observation.reshape(1, 14)

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

    # x 轴
    frame_start_postition, frame_posture = p.getBasePositionAndOrientation(Env.target)
    frame_start_postition = [0, 0.8, 0.3]
    R_Mat = np.array(p.getMatrixFromQuaternion(frame_posture)).reshape(3, 3)
    x_axis = R_Mat[:, 0]
    x_end_p = (np.array(frame_start_postition) + np.array(x_axis * 5)).tolist()
    x_line_id = p.addUserDebugLine(frame_start_postition, x_end_p, [1, 0, 0])

    # y 轴
    y_axis = R_Mat[:, 1]
    y_end_p = (np.array(frame_start_postition) + np.array(y_axis * 5)).tolist()
    y_line_id = p.addUserDebugLine(frame_start_postition, y_end_p, [0, 1, 0])

    # z轴
    z_axis = R_Mat[:, 2]
    z_end_p = (np.array(frame_start_postition) + np.array(z_axis * 5)).tolist()
    z_line_id = p.addUserDebugLine(frame_start_postition, z_end_p, [0, 0, 1])

    # time.sleep(10)
    check_env(Env, warn=True)

    for i in range(100):
        p.stepSimulation()
    Env.render()
    print("test going")
    time.sleep(10)
    # observation, reward, terminated, truncated, info = Env.step([0,0,0,0,0,20])
    # print(reward)
