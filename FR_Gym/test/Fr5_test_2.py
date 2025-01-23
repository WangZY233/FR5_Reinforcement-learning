import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import tensorflow as tf
from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from FR_Gym import FR5_Env
import time
from utils.arguments import get_args

if __name__ == '__main__':
    args, kwargs = get_args()
    env = FR5_Env(gui=False)
    env.render()

    model_dir = "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/FR5_Reinforcement-learning/models/PPO/0122-145712/"
    success_rate = []

    # 创建TensorBoard的SummaryWriter对象
    log_dir = "./test_logs/"
    success_rate = []
    writer = SummaryWriter(log_dir)
    model_0 = PPO.load("/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/pick_model")
    model_1 = PPO.load(
        "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/place_model")
    model_2 = PPO.load(
        "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/button_model")
    model_3 = PPO.load(
        "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/catch_model")
    model_4 = PPO.load(
        "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/models/trans_model")
    guide_model = [model_0, model_1, model_2, model_3, model_4]
    for episode in range(1001):
        model_path = os.path.join(model_dir, f"PPO-run-eposide{episode}.zip")
        if not os.path.exists(model_path):
            continue
        model_path = os.path.splitext(model_path)[0]
        success_rate_episode = []
        model = PPO.load(model_path)
        for test_stage in range(5):
            test_num = args.test_num  # 测试次数
            success_num = 0  # 成功次数
            print("测试次数：", test_num)
            for i in range(test_num):
                env.stage = 0
                for j in range(test_stage):
                    # time.sleep(1)
                    done = False
                    score = 0
                    # time.sleep(3)
                    step = 0
                    env.stage = j
                    state, _ = env.reset()
                    while not done:
                        step += 1
                        action, _ = guide_model[j].predict(observation=env.guide_observation, deterministic=True)
                        state, reward, done, _, info = env.step(action=action)
                        score += reward
                        # time.sleep(0.02)
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
                    # time.sleep(0.02)
                    if info['is_success']:
                        success_num += 1
                print("奖励：", score)
            success_rate_episode.append(success_num / test_num)
            print("阶段：", test_stage, "成功率：", success_rate[test_stage])
        success_rate.append(success_rate_episode)
        print("模型：", episode,"成功率：", success_rate[-1])
        # 将成功率写入TensorBoard
        for i, rate in enumerate(success_rate_episode):
            writer.add_scalar(f'Success_Rate/Stage_{i}', rate, episode)
        #

    print("成功率：", success_rate)
    env.close()

    # 关闭TensorBoard的SummaryWriter对象
    writer.close()
