import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from stable_baselines3 import A2C,PPO,DDPG,TD3,HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from Fr5_env import FR5_Env
import time
import os
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

now = time.strftime('%m%d-%H%M%S', time.localtime())
models_dir = f"F:\\Pycharm_project\\RL\\models\\TD3\\"+now
logs_dir = f"F:\\Pycharm_project\\RL\\logs\\TD3\\"+now
checkpoints = f"F:\\Pycharm_project\\RL\\checkpoints\\TD3\\"+now


def evaluate_model(model, env, n_eval_episodes=10):
    success_count = 0
    episode_rewards = []
    episode_len = []

    for episode in range(n_eval_episodes):
        episode_reward = 0
        obs,_ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, reward, done, _,info = env.step(action)
            episode_reward += reward

            if done and info.get('is_success'):
                success_count += 1

        episode_len.append(info.get('step_num'))
        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    episode_len_mean = np.mean(episode_len)
    std_reward = np.std(episode_rewards)
    success_rate = success_count / n_eval_episodes

    return mean_reward, std_reward, success_rate, episode_len_mean


def make_env():
    def _init():
        env = FR5_Env()
        env = Monitor(env, logs_dir)
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
    num_train = 1
    env = SubprocVecEnv([make_env() for i in range(num_train)])
    # env = FR5_Env()
    new_logger = configure(logs_dir, ["stdout", "csv", "tensorboard"])

    # Define and Train the agent
    # model = TD3(policy = "MlpPolicy",
    #             env = env,
    #             learning_rate = 0.001,
    #             buffer_size = 1000000,
    #             learning_starts = 100,
    #             batch_size = 2048,
    #             tau = 0.005,
    #             gamma = 0.99,
    #             train_freq = (5, "step"),
    #             gradient_steps = -1,
    #             action_noise = None,
    #             replay_buffer_class = None,
    #             replay_buffer_kwargs = None,
    #             optimize_memory_usage = False,
    #             policy_delay = 2,
    #             target_policy_noise = 0.2,
    #             target_noise_clip = 0.5,
    #             stats_window_size = 100,
    #             tensorboard_log = logs_dir,
    #             policy_kwargs= None,
    #             verbose = 1,
    #             seed = None,
    #             device = "cuda",
    #             _init_setup_model =  True)
    model = TD3.load("F:\\Pycharm_project\\RL\\models\\TD3\\1113-114547\\DDPG-run-eposide74.zip",env=env,print_system_info=True)
    model.set_logger(new_logger)

    # 创建测试环境
    eval_env = FR5_Env()
    eval_env = Monitor(eval_env, logs_dir)
    eval_env.render()

    TIMESTEPS = 5000
    for eposide in range(1000):
        # 创建 CheckpointCallback 实例来保存模型检查点
        checkpoint_callback = CheckpointCallback(save_freq=100, save_path=checkpoints)
        model.learn(total_timesteps=TIMESTEPS,
                    tb_log_name=f"DDPG-run-eposide{eposide}", # TensorBoard 日志运行的名称
                    reset_num_timesteps=False,  # 是否重置模型的当前时间步数
                    callback=CheckpointCallback,  # 在每一步调用的回调，可以用CheckpointCallback来创建一个存档点和规定存档间隔。
                    log_interval=5  #  记录一次信息的时间步数
                    )
        
         # 测试模型
        mean_reward, std_reward, success_rate, episode_len_mean= evaluate_model(model, eval_env, n_eval_episodes=10)
        # logger记录测试结果
        model.logger.record("eval/mean_reward", mean_reward)
        model.logger.record("eval/std_reward", std_reward)
        model.logger.record('eval/success_rate', success_rate)
        model.logger.record('eval/episode_len_mean', episode_len_mean)
        model.logger.dump(step=eposide)

        model.save(models_dir+f"/TD3-run-eposide{eposide}")
        print(f"**************eposide--{eposide} saved**************")
