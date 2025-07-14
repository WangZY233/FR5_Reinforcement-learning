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
    env = FR5_Env(gui=True, guide_rate=0)
    env.render()
    model = PPO.load(
        "/home/dianrobot/PycharmProjects/FR5_Reinforcement-learning/FR_Gym/FR5_Reinforcement-learning/models/PPO/0516-144637/best_model.zip")
    model_0 = PPO.load(
        "/home/dianrobot/PycharmProjects/FR5_Reinforcement-learning/models/pick_model")
    model_1 = PPO.load(
        "/home/dianrobot/PycharmProjects/FR5_Reinforcement-learning/models/place_model")
    model_2 = PPO.load(
        "/home/dianrobot/PycharmProjects/FR5_Reinforcement-learning/models/button_model")
    model_3 = PPO.load(
        "/home/dianrobot/PycharmProjects/FR5_Reinforcement-learning/models/catch_model")
    model_4 = PPO.load(
        "/home/dianrobot/PycharmProjects/FR5_Reinforcement-learning/models/trans_model")
    guide_model = [model_0, model_1, model_2, model_3, model_4]
    success_rate = []

    # for test_stage in range(5):
    #     test_stage = 4
    #     info = {"is_success": True}
    #     test_num = args.test_num  # 测试次数
    #     test_num = 10  # 测试次数
    #     success_num = 0  # 成功次数
    #     error_num = 0  # 前置任务失败次数
    #     print("测试次数：", test_num)
    #     for i in range(test_num):
    #         env.stage = 0
    #         info = {"is_success": True}
    #         for j in range(test_stage):
    #             # time.sleep(1)
    #             done = False
    #             score = 0
    #             # time.sleep(3)
    #             step = 0
    #             env.stage = j
    #             state, _ = env.reset()
    #             while not done:
    #                 step += 1
    #                 action, _ = guide_model[j].predict(observation=env.guide_observation, deterministic=True)
    #                 state, reward, done, _, info = env.step(action=action)
    #                 score += reward
    #                 time.sleep(0.02)
    #         if info['is_success']:
    #             done = False
    #             score = 0
    #             # time.sleep(3)
    #             step = 0
    #             env.stage = test_stage
    #             state, _ = env.reset()
    #             while not done:
    #                 step += 1
    #                 action, _ = model.predict(observation=state, deterministic=True)
    #                 state, reward, done, _, info = env.step(action=action)
    #                 score += reward
    #                 time.sleep(0.02)
    #                 if info['is_success']:
    #                     success_num += 1
    #             print("奖励：", score/step)
    #         else:
    #             print("前置任务失败！不计入总次数")
    #             error_num += 1
    #     if test_num - error_num == 0:
    #         success_rate.append(0)
    #     else:
    #         success_rate.append(success_num / (test_num - error_num))
    #     print("阶段：", test_stage, "成功率：", success_rate[test_stage])
    # print("成功率：", success_rate)
    # 不使用指导模型
    test_num = args.test_num  # 测试次数
    test_num = 10  # 测试次数
    for i in range(test_num):
        for test_stage in range(5):
            success_num = 0  # 成功次数
            done = False
            score = 0
            # time.sleep(3)
            step = 0
            env.stage = test_stage
            state, _ = env.reset()
            while not done:
                step += 1
                action, _ = model.predict(observation=state, deterministic=True)
                state, reward, done, _, info = env.step(action=action)
                score += reward
                time.sleep(0.005)
                if info['is_success']:
                    if test_stage == 4:
                        success_num += 1
                        print("成功！")
    env.close()
