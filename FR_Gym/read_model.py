import torch

# 加载.pth文件
model = torch.load('/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_0/FR_Gym/2_2/models/PPO/1010-095300/best_model/policy.pth')

# 查看模型结构
print(model)

# 查看模型参数
for name, param in model.named_parameters():
    print(name, param)