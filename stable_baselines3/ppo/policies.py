# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, ActorCriticLSTMPolicy, MultiInputActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
LSTMPolicy = ActorCriticLSTMPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
