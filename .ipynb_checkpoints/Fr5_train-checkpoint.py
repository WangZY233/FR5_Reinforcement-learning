import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from stable_baselines3 import A2C,PPO,DDPG,TD3
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from Fr5_env import FR5_Env
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback,CallbackList,BaseCallback,CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy  
from loguru import logger

now = time.strftime('%m%d-%H%M%S', time.localtime())
models_dir = f"F:\\Pycharm_project\\RL\\models\\PPO\\"+now
logs_dir = f"F:\\Pycharm_project\\RL\\logs\\PPO\\"+now
checkpoints = f"F:\\Pycharm_project\\RL\\checkpoints\\PPO\\"+now
test = f"F:\\Pycharm_project\\RL\\logs\\test\\"+now


def evaluate_model(model, env, n_eval_episodes=10):
    success_count = 0
    episode_rewards = []

    for episode in range(n_eval_episodes):
        episode_reward = 0
        obs,_ = env.reset_2()
        done = False

        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, reward, done, _,info = env.step_2(action)
            episode_reward += reward

            if done and info.get('is_success', True):
                success_count += 1

        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = success_count / n_eval_episodes

    return mean_reward, std_reward, success_rate

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.n_envs = 16  # 假设有2个环境
        self.episode_rewards = [0.0 for _ in range(self.n_envs)]
        self.episode_lengths = [0 for _ in range(self.n_envs)]
        self.episode_counts = [0 for _ in range(self.n_envs)]
        self.log_interval = 5  # 每100个回合记录一次

    def _on_step(self) -> bool:
        # 遍历所有环境
        for i in range(len(self.locals['rewards'])):
            self.episode_rewards[i] += self.locals['rewards'][i]
            self.episode_lengths[i] += 1

            # 检查回合是否结束
            if self.locals['dones'][i]:
                self.episode_counts[i] += 1

                # 每100个回合记录一次平均奖励
                if self.episode_counts[i] % self.log_interval == 0:
                    avg_reward = self.episode_rewards[i] / 5
                    self.model.logger.record(f"reward/env_{i}", avg_reward, exclude="stdout")
                    self.model.logger.dump(step=self.episode_counts[i])

                    # 重置累积奖励和回合长度
                    self.episode_rewards[i] = 0.0
                    self.episode_lengths[i] = 0

        return True



def make_env():
    def _init():
        env = FR5_Env()
        env = Monitor(env, logs_dir)
        env.render()
        env.reset()
        return env
    set_random_seed(0)
    return _init

if __name__ == '__main__':
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):    
        os.makedirs(logs_dir)
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)

    # Instantiate the env
    num_train = 16
    env = SubprocVecEnv([make_env() for i in range(num_train)])
    # env = DummyVecEnv([make_env() for i in range(num_train)])
    
    
    # env = FR5_Env()
    
    new_logger = configure(logs_dir, ["stdout", "csv", "tensorboard"])
    # Define and Train the agent
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir,batch_size=256,device="cuda")
    model = PPO.load("F:\\Pycharm_project\\RL\\models\\PPO\\1211-140713\\pick_model.zip",env=env,print_system_info=True)
    # model = TD3.load("F:\\Pycharm_project\\RL\\models\\DDPG\\1031-160549\\DDPG-run-eposide1.zip",env=env)
    model.set_logger(new_logger)
    tensorboard_callback = CustomCallback()
    

    # 创建测试环境
    # eval_env = ([make_env() for i in range(num_train)])
    eval_env = FR5_Env()
    eval_env = Monitor(eval_env, logs_dir)
    eval_env.render()
    

    eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                             log_path=logs_dir, eval_freq=3000,
                             deterministic=True, render=True,n_eval_episodes = 100)

    TIMESTEPS = 30000
    for eposide in range(1000):
        # 创建 CheckpointCallback 实例来保存模型检查点
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoints)
        model.learn(total_timesteps=TIMESTEPS,
                    tb_log_name=f"PPO-run-eposide{eposide}", # TensorBoard 日志运行的名称
                    reset_num_timesteps=False,  # 是否重置模型的当前时间步数
                    callback=CallbackList([eval_callback,tensorboard_callback]),  # 在每一步调用的回调，可以用CheckpointCallback来创建一个存档点和规定存档间隔。
                    log_interval=10  #  记录一次信息的时间步数
                    )
        
        
        # # # 测试模型
        # mean_reward, std_reward, success_rate = evaluate_model(model, eval_env, n_eval_episodes=100)
        # # logger记录测试结果
        # model.logger.record("eval/mean_reward", mean_reward)
        # model.logger.record("eval/std_reward", std_reward)
        # model.logger.record('eval/success_rate', success_rate)
        # model.logger.dump(step=eposide)
        # 保存模型
        model.save(models_dir+f"/PPO-run-eposide{eposide}")
        logger.info(f"**************eposide--{eposide} saved**************")
