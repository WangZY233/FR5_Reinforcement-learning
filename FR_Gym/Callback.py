from stable_baselines3.common.callbacks import EvalCallback,CallbackList,BaseCallback,CheckpointCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.n_envs = 16  # 假设有2个环境
        
        self.episode_lengths = [[0 for _ in range(self.n_envs)] for __ in range(5)]
        self.episode_counts = [[0 for _ in range(self.n_envs)]for __ in range(5)]

        self.episode_total_rewards = [[0.0 for _ in range(self.n_envs)] for __ in range(5)]
        self.episode_pose_rewards = [[0.0 for _ in range(self.n_envs)] for __ in range(5)]
        self.episode_dis_rewards = [[0.0 for _ in range(self.n_envs)] for __ in range(5)]
        self.episode_success = [[0.0 for _ in range(self.n_envs)] for __ in range(5)]

        self.log_interval = 30  # 每30个回合记录一次

    def _on_step(self) -> bool:
        # 遍历所有环境
        for i in range(len(self.locals['rewards'])):
            self.episode_total_rewards[self.locals['infos'][i]['stage']][i] += self.locals['rewards'][i]
            self.episode_pose_rewards[self.locals['infos'][i]['stage']][i] += self.locals['infos'][i]['pose_reward']
            self.episode_dis_rewards[self.locals['infos'][i]['stage']][i] += self.locals['infos'][i]['distance_reward']
            self.episode_success[self.locals['infos'][i]['stage']][i] += self.locals['infos'][i]['success_reward']
            self.episode_lengths[self.locals['infos'][i]['stage']][i] += 1

            # 检查回合是否结束
            if self.locals['dones'][i]:
                self.episode_counts[self.locals['infos'][i]['stage']][i] += 1

                # 每100个回合记录一次平均奖励
                if self.episode_counts[self.locals['infos'][i]['stage']][i] % self.log_interval == 0:
                    avg_reward = self.episode_total_rewards[self.locals['infos'][i]['stage']][i] / self.log_interval
                    avg_pose_reward = self.episode_pose_rewards[self.locals['infos'][i]['stage']][i] / self.log_interval
                    avg_dis_reward = self.episode_dis_rewards[self.locals['infos'][i]['stage']][i] / self.log_interval
                    avg_success = self.episode_success[self.locals['infos'][i]['stage']][i] / self.log_interval

                    self.model.logger.record(f"stage{self.locals['infos'][i]['stage']}/reward/env_{i}", avg_reward, exclude="stdout")
                    self.model.logger.record(f"stage{self.locals['infos'][i]['stage']}/pose_reward/env_{i}", avg_pose_reward, exclude="stdout")
                    self.model.logger.record(f"stage{self.locals['infos'][i]['stage']}/distance_reward/env_{i}", avg_dis_reward, exclude="stdout")
                    self.model.logger.record(f"stage{self.locals['infos'][i]['stage']}/success_rate/env_{i}", avg_success, exclude="stdout")

                    self.model.logger.dump(step=self.episode_counts[self.locals['infos'][i]['stage']][i]/(self.log_interval-1))

                    # 重置累积奖励和回合长度
                    self.episode_total_rewards[self.locals['infos'][i]['stage']][i] = 0.0
                    self.episode_pose_rewards[self.locals['infos'][i]['stage']][i] = 0.0
                    self.episode_dis_rewards[self.locals['infos'][i]['stage']][i] = 0.0
                    self.episode_success[self.locals['infos'][i]['stage']][i] = 0.0
                    self.episode_lengths[self.locals['infos'][i]['stage']][i] = 0

        return True