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

def cal_success_reward(self,distance):
    '''
        计算成功/失败奖励
    '''
    # 1.计算碰撞奖励
    # 若机械臂成功抓取目标，那么任务成功
    # 若机械臂发生其他碰撞（桌子或其他关节碰撞目标），那么任务失败
    gripper_joint_indices = [8,9]
    target_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.target)
    table_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.table)
    self_targettable_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.targettable)

    # 定义碰撞变量
    gripper_contact = False
    other_contact = False
    target_contact = False
    targettable_contact = False

    # 机械臂关节碰撞目标
    for contact_point in target_contact_points:
        link_index = contact_point[3]
        if link_index == 8 or link_index == 9:
            gripper_contact = True
            # logger.info("机械臂抓夹接触目标!")
        # 检查是否有非夹爪关节接触目标
        if link_index not in gripper_joint_indices and gripper_contact == False:
            target_contact = True
            # logger.info("机械臂接触目标！")
    
    # 碰撞桌子
    for contact_point in table_contact_points:
        link_index = contact_point[3]
        if not(link_index == 0 or link_index == 1):
            other_contact = True
            table_contact = True
            # logger.info("碰撞桌子！")
    
    # 碰撞目标杯子的台子
    for contact_point in self_targettable_contact_points:
        other_contact = True
        targettable_contact = True
        # logger.info("碰撞目标杯子的台子! ")

    # all:判断成功或失败
    # 并计算成功或失败的奖励

    success_reward = 0
    # 夹爪中心和目标之间距离小于一定值，则任务成功
    if self.success == True and self.step_num <= 100:
        success_reward = 1000
        self.terminated = True
        self.success = True
        logger.info("成功抓取！！！！！！！！！！执行步数：%s  距离目标:%s"%(self.step_num, distance))
        # self.truncated = True

    # 碰撞桌子，或者碰撞自身，或者碰撞台子
    elif other_contact:
        success_reward = - 100
        self.terminated = True
        if targettable_contact == True:
            logger.info("失败！碰撞目标杯子的台子! 执行步数：%s    距离目标:%s"%(self.step_num, distance))
        else:
            logger.info("失败！碰撞桌子！ 执行步数：%s    距离目标:%s"%(self.step_num, distance))
        # self.truncated = True

    # 机械臂关节接触目标
    elif target_contact:
        success_reward = - 100
        self.terminated = True
        logger.info("失败！机械臂接触目标！ 执行步数：%s    距离目标:%s"%(self.step_num, distance))
    
    # 机械臂夹爪接触目标
    elif gripper_contact:
        success_reward = - 80
        self.terminated = True
        logger.info("失败！机械臂抓夹接触目标!  执行步数：%s    距离目标:%s"%(self.step_num, distance))

    # 机械臂执行步数过多
    elif self.step_num > 100:
        success_reward = - 100
        self.terminated = True
        logger.info("失败！执行步数过多！ 执行步数：%s    距离目标:%s"%(self.step_num, distance))
    
    return success_reward

def cal_dis_reward(self,distance):
    '''计算距离奖励'''
    if self.step_num == 0:
        distance_reward = 0
    else:
        # if distance <= 0.41:
        distance_reward = 1000*(self.distance_last-distance)
        # elif distance > 0.41:
        #     distance_reward = -2 * pow(math.e, 4.3*distance)
    # logger.debug("相对距离：%f"%(self.distance_last-distance))
    # logger.debug("距离奖励:%f"%distance_reward)
    # 保存上一次的距离
    self.distance_last = distance
    return distance_reward

def cal_pose_reward(self):
    '''姿态奖励'''
    # 计算夹爪的朝向
    gripper_orientation = p.getLinkState(self.fr5, 7)[1]
    gripper_orientation = R.from_quat(gripper_orientation)
    gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)
    # print("夹爪旋转角度： ",gripper_orientation)
    # 计算夹爪的姿态奖励
    pose_reward = -(pow(gripper_orientation[0]+90, 2) + pow(gripper_orientation[1], 2) + pow(gripper_orientation[2], 2))
    # logger.debug("姿态奖励：%f"%pose_reward)
    return pose_reward*0.01

def grasp_reward(self):
    '''获取奖励'''
    info = {}
    total_reward = 0

    distance = get_distance(self)
    judge_success(self,distance,success_dis=0.02)

    # 计算奖励
    success_reward = cal_success_reward(self,distance)
    distance_reward = cal_dis_reward(self,distance)
    pose_reward = cal_pose_reward(self)

    total_reward = success_reward + pose_reward + distance_reward
    
    self.truncated = False
    self.reward = total_reward
    info['reward'] = self.reward
    info['is_success'] = self.success
    info['step_num'] = self.step_num

    info['success_reward'] = (1 if self.success else 0)
    info['distance_reward'] = distance_reward
    info['pose_reward'] = pose_reward

    return total_reward,info

def judge_success(self,distance,success_dis):
    '''判断成功或失败'''
    if distance < success_dis:
        self.success = True
    else:
        self.success = False
        # total_reward = total_reward + (0.3 - distance)

def get_distance(self):
    '''判断机械臂与夹爪的距离'''
    Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
    Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
    Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
    relative_position = np.array([0, 0, 0.15])
    # 固定夹爪相对于机械臂末端的相对位置转换
    rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
    rotated_relative_position = rotation.apply(relative_position)
    gripper_centre_pos = [Gripper_posx, Gripper_posy,Gripper_posz] + rotated_relative_position
    self.target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])
    distance = math.sqrt((gripper_centre_pos[0]-self.target_position[0])**2+(gripper_centre_pos[1]-self.target_position[1])**2+(gripper_centre_pos[2]-self.target_position[2])**2)
    # logger.debug("distance:%s"%str(distance))
    return distance