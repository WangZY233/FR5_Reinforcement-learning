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
from interval import Interval


def cal_success_reward(self, distance):
    success_reward = 0
    # 夹爪中心和目标之间距离小于一定值，则任务成功
    if self.success == True and self.step_num <= 100:
        success_reward = 1
        self.terminated = True
        self.success = True
        logger.info("成功抓取！！！！！！！！！！当前阶段:%s  执行步数：%s  距离目标:%s" % (self.stage, self.step_num, distance))
        if self.stage != 4:
            self.stage += 1
        else:
            self.stage = 1
        # self.truncated = True

    # 机械臂执行步数过多
    if self.step_num > 100:
        success_reward = - 1
        self.terminated = True
        logger.info("失败！执行步数过多！当前阶段：%s 执行步数：%s    距离目标:%s" % (self.stage, self.step_num, distance))
        self.stage = 1

    return success_reward


def cal_pose_reward(self):
    '''姿态奖励'''
    # 计算夹爪的朝向
    gripper_orientation = p.getLinkState(self.fr5, 7)[1]
    gripper_orientation = R.from_quat(gripper_orientation)
    gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)
    # 计算夹爪的姿态奖励
    pose_reward = -(
            pow(gripper_orientation[0] + 90, 2) + pow(gripper_orientation[1], 2) + pow(gripper_orientation[2], 2))
    # logger.debug("姿态奖励：%f"%pose_reward)
    return pose_reward * 0.01


def grasp_reward(self, diff=0):
    '''获取奖励'''
    info = {}
    # stage需要在结算之前记录
    info['stage'] = self.stage
    total_reward = 0

    distance = get_distance(self)
    pose_reward = cal_pose_reward(self)
    real_distance = get_real_distance(self)
    judge_success(self, distance, pose_reward, success_dis=0.015, success_pose=-100)

    # 计算奖励
    success_reward = cal_success_reward(self, distance)
    # 现有模型与目标模型的差异惩罚
    diff_reward = -diff/10
    total_reward = success_reward + diff_reward

    self.truncated = False
    self.reward = total_reward
    info['reward'] = self.reward
    info['is_success'] = self.success
    info['step_num'] = self.step_num

    info['success_reward'] = (1 if self.success else 0)
    info['distance_reward'] = diff_reward
    info['pose_reward'] = pose_reward

    return total_reward, info


def judge_success(self, distance, pose, success_dis, success_pose):
    '''判断成功或失败'''
    if distance < success_dis:
        if pose > success_pose:
            self.success = True
        else:
            self.success = False
    else:
        self.success = False
        # total_reward = total_reward + (0.3 - distance)


def get_distance(self):
    '''判断机械臂与夹爪的距离'''
    Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
    Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
    Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
    if self.stage == 2:
        relative_position = np.array([0, 0, 0.183])
    else:
        relative_position = np.array([0, 0, 0.15])
    # 固定夹爪相对于机械臂末端的相对位置转换
    rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
    rotated_relative_position = rotation.apply(relative_position)
    gripper_centre_pos = [Gripper_posx, Gripper_posy, Gripper_posz] + rotated_relative_position
    self.target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])
    distance = math.sqrt((gripper_centre_pos[0] - self.goalx) ** 2 +
                         ((gripper_centre_pos[1] - self.goaly) ** 2) +
                         (gripper_centre_pos[2] - self.goalz) ** 2)
    # logger.debug("distance:%s"%str(distance))
    return distance


def her_reward(self):
    #获取夹爪当前位置：
    Gripper_pos = p.getLinkState(self.fr5, 6)[0]

    if self.stage == 2:
        relative_position = np.array([0, 0, 0.183])
    else:
        relative_position = np.array([0, 0, 0.15])
    # 固定夹爪相对于机械臂末端的相对位置转换
    rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
    rotated_relative_position = rotation.apply(relative_position)
    gripper_centre_pos = Gripper_pos + rotated_relative_position
    #


def get_real_distance(self):
    '''判断机械臂与夹爪的距离'''
    Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
    Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
    Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
    relative_position = np.array([0, 0, 0.15])
    # 固定夹爪相对于机械臂末端的相对位置转换
    rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
    rotated_relative_position = rotation.apply(relative_position)
    gripper_centre_pos = [Gripper_posx, Gripper_posy, Gripper_posz] + rotated_relative_position
    self.target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])
    distance = math.sqrt((gripper_centre_pos[0] - self.target_position[0]) ** 2 +
                         (gripper_centre_pos[1] - self.target_position[1]) ** 2 +
                         (gripper_centre_pos[2] - self.target_position[2]) ** 2)
    # logger.debug("distance:%s"%str(distance))
    return distance
