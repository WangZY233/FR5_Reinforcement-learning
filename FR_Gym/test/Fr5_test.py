'''
@Author: Prince Wang 
@Date: 2024-02-22 
@Last Modified by:   Prince Wang 
@Last Modified time: 2023-10-24 23:04:04 
'''
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from stable_baselines3 import A2C, PPO, DDPG, TD3
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from FR_Gym import FR5_Env
import time
from utils.arguments import get_args

if __name__ == '__main__':
    args, kwargs = get_args()
    env = FR5_Env(gui=True)
    env.render()
    # model = PPO.load("/home/wangzy/FR5_Reinforcement-learning-long_sequence/FR5_Reinforcement-learning/models/PPO/1224-175545/pick_model.zip")
    model_0 = PPO.load("/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/pick_model")
    model_1 = PPO.load(
            "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/place_model")
    model_2 = PPO.load(
            "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/button_model")
    model_3 = PPO.load(
            "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/catch_model")
    model_4 = PPO.load(
            "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/trans_model")
    # model_longSequence = PPO.load("/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/FR_Gym/FR5_Reinforcement-learning/models/PPO/1219-013116/PPO-run-eposide540.zip")
    # model = TD3.load("F:\\Pycharm_project\\RL\\models\\TD3\\TD3-run-eposide270.zip")
    # model = DDPG.load("F:\\Pycharm_project\\RL\\models\\DDPG\\DDPG-run-eposide282.zip")
    test_num = args.test_num  # 测试次数
    success_num = 0  # 成功次数
    print("测试次数：", test_num)
    for i in range(test_num):

        time.sleep(1)
        done = False
        score = 0
        # time.sleep(3)
        step = 0
        env.stage = 0
        state, _ = env.reset()
        while not done:
            step += 1
            # action = env.action_space.sample()     # 随机采样动作
            print("state:", state)
            action, _ = model_0.predict(observation=env.guide_observation, deterministic=True)

            # print("action:",action)
            # if step % 40 == 0:
            #     env.moveTarget()
            state, reward, done, _, info = env.step(action=action)

            score += reward
            # env.render()
            # print("state:", state)

            time.sleep(0.02)
        env.stage = 1
        state, _ = env.reset()
        time.sleep(1)
        done = False
        score = 0
        # time.sleep(3)
        step = 0
        while not done:
            step += 1
            # action = env.action_space.sample()     # 随机采样动作
            # print("state:", state)
            action, _ = model_1.predict(observation=env.guide_observation, deterministic=True)
        
            # print("action:",action)
            # if step % 40 == 0:
            #     env.moveTarget()
            state, reward, done, _, info = env.step(action=action)
        
            score += reward
            # env.render()
            # print("state:", state)
        
            time.sleep(0.02)
        env.stage = 2
        env.reset()
        done = False
        score = 0
        # time.sleep(3)
        step = 0
        while not done:
            step += 1
            # action = env.action_space.sample()     # 随机采样动作
            # print("state:", state)
            action, _ = model_2.predict(observation=env.guide_observation, deterministic=True)
        
            # print("action:",action)
            # if step % 40 == 0:
            #     env.moveTarget()
            state, reward, done, _, info = env.step(action=action)
        
            score += reward
            # env.render()
            # print("state:", state)
        
            time.sleep(0.02)
        '''重置环境参数'''
        env.stage = 3
        env.reset()
        done = False
        score = 0
        # time.sleep(3)
        step = 0
        while not done:
            step += 1
            # action = env.action_space.sample()     # 随机采样动作
            # print("state:", state)
            action, _ = model_3.predict(observation=env.guide_observation, deterministic=True)
        
            # print("action:",action)
            # if step % 40 == 0:
            #     env.moveTarget()
            state, reward, done, _, info = env.step(action=action)
        
            score += reward
            # env.render()
            # print("state:", state)
        
            time.sleep(0.02)
        env.stage = 4
        env.reset()
        done = False
        score = 0
        # time.sleep(3)
        step = 0
        while not done:
            step += 1
            # action = env.action_space.sample()     # 随机采样动作
            # print("state:", state)
            action, _ = model_4.predict(observation=env.guide_observation, deterministic=True)
        
            # print("action:",action)
            # if step % 40 == 0:
            #     env.moveTarget()
            state, reward, done, _, info = env.step(action=action)
        
            score += reward
            # env.render()
            # print("state:", state)
        
            time.sleep(0.02)
        time.sleep(1)

        if info['is_success']:
            success_num += 1
        print("奖励：", score)
    success_rate = success_num / test_num
    print("成功率：", success_rate)
    env.close()
