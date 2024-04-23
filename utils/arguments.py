'''
Author: wangziyuan 13536655301
Date: 2024-04-10 22:55:27
LastEditors: wangziyuan 13536655301
LastEditTime: 2024-04-23 15:04:49
'''

import argparse
import time
now = time.strftime('%m%d-%H%M%S', time.localtime())

def get_args():
    parser = argparse.ArgumentParser(description="Running time configurations")
    
    parser.add_argument('--model_path', type=str, default="FR5_Reinforcement-learning\models\PPO\best_model_1.5mm.zip")
    parser.add_argument('--test_num', type=int, default=100)
    parser.add_argument('--gui', type=bool, default=False)
    parser.add_argument('--models_dir', type=str, default=f"models\\PPO\\"+now)
    parser.add_argument('--logs_dir', type=str, default=f"logs\\PPO\\"+now)
    parser.add_argument('--checkpoints', type=str, default=f"checkpoints\\PPO\\"+now)
    parser.add_argument('--test', type=str, default=f"logs\\test\\"+now)
    parser.add_argument('--timesteps', type=int, default=30000)


    args = parser.parse_args()

    kwargs = vars(args)

    return args, kwargs