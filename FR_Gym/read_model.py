from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboardX import SummaryWriter
tfevent_filename1 = r"/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/test_logs/0123-223501/1000/events.out.tfevents.1737773622.Woshihg"
tfevent_filename2 = r"/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/test_logs/0124-231412/0125-120352/events.out.tfevents.1737777832.Woshihg"
tfevent_filename3 = r"/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_longSequence/test_logs/0124-231412/0125-124409/events.out.tfevents.1737780249.Woshihg"
tensorboard_logdir = "./1500"
writer = SummaryWriter(tensorboard_logdir)
# 初始化字典，存储每个阶段的成功率，索引从0、5、10到995
success_rate = {}
for i in range(0, 1000, 5):
    success_rate[i] = []
for event in summary_iterator(tfevent_filename1):
    for value in event.summary.value:
        writer.add_scalar(value.tag, value.simple_value, event.step)

for event in summary_iterator(tfevent_filename2):
    for value in event.summary.value:
        writer.add_scalar(value.tag, value.simple_value, event.step)

for event in summary_iterator(tfevent_filename3):
    for value in event.summary.value:
        writer.add_scalar(value.tag, value.simple_value, event.step)

writer.close()

# [[1.0, 0.7, 0.0, 0.04, 0.0], [1.0, 0.68, 0.0, 0.02, 0.0], [0.94, 0.64, 0.02, 0.04, 0.0],
# [0.94, 0.78, 0.0, 0.0, 0.0], [0.94, 0.6, 0.0, 0.02, 0.0], [1.0, 0.82, 0.0, 0.0, 0.02],
# [1.0, 0.8, 0.04,, 0.02], [0.94, 0.64, 0.02, 0.0, 0.0], [1.0, 0.62, 0.02, 0.0, 0.0],
# [1.0, 0.86, 0.02, 0.0, 0.04], [1.0, 0.64, 0.0, 0.0, 0.08], [0.96, 0.78, 0.0, 0.0, 0.04],
# [0.98, 0.62, 0.02, 0.06, 0.0], [0.98, 0.3, 0.0, 0.0, 0.08], [0.96, 0.66, 0.02, 0.0, 0.02],
# [0.92, 0.58, 0.0, 0.0, 0.06], [0.94, 0.72, 0.04, 0.0, 0.02], [0.98, 0.56, 0.0, 0.02, 0.02],
# [0.5, 0.34, 0.0, 0.0, 0.0], [1.0, 0.7, 0.0, 0.04, 0.0], [0.96, 0.62, 0.0, 0.0, 0.02],
# [0.76, 0.78, 0.08, 0.0, 0.02], [0.88, 0.8, 0.06, 0.04, 0.0], [0.98, 0.48, 0.06, 0.08, 0.0],
# [1.0, 0.82, 0.06, 0.0, 0.0], [1.0, 0.96, 0.0, 0.06, 0.0], [0.96, 0.58, 0.08, 0.04, 0.0],
# [0.96, 0.84, 0.02, 0.02, 0.0], [0.98, 0.72, 0.02, 0.04, 0.0], [1.0, 0.82, 0.02, 0.0, 0.0],
# [0.9, 0.8, 0.0, 0.02, 0.0], [1.0, 0.78, 0.0, 0.08, 0.0], [0.92, 0.76, 0.0, 0.04, 0.0],
# [1.0, 0.76, 0.02, 0.24, 0.02], [0.96, 0.74, 0.02, 0.14, 0.0], [1.0, 0.38, 0.04, 0.12, 0.0],
# [1.0, 0.76, 0.12, 0.08, 0.0], [1.0, 0.84, 0.02, 0.34, 0.0], [0.96, 0.58, 0.1, 0.26, 0.0],
# [0.98, 0.62, 0.08, 0.16, 0.0], [0.96, 0.44, 0.14, 0.26, 0.0], [0.98, 0.64, 0.18, 0.36, 0.0],
# [0.94, 0.7, 0.28, 0.28, 0.0], [0.92, 0.78, 0.18, 0.3, 0.0], [0.94, 0.72, 0.34, 0.3, 0.0],
# [1.0, 0.88, 0.24, 0.46, 0.0], [0.98, 0.8, 0.26, 0.4, 0.0], [0.96, 0.78, 0.2, 0.12, 0.0],
# [0.96, 0.56, 0.34, 0.36, 0.0], [0.98, 0.64, 0.3, 0.1, 0.0], [0.92, 0.78, 0.16, 0.62, 0.0],
# [0.96, 0.64, 0.28, 0.66, 0.0], [0.92, 0.62, 0.36, 0.42, 0.0], [0.96, 0.58, 0.36, 0.42, 0.0],
# [0.96, 0.62, 0.14, 0.44, 0.0], [0.94, 0.62, 0.18, 0.44, 0.0], [0.94, 0.36, 0.1, 0.52, 0.0],
# [0.98, 0.54, 0.18, 0.62, 0.0], [1.0, 0.42, 0.12, 0.36, 0.0], [1.0, 0.46, 0.02, 0.28, 0.0],
# [0.96, 0.56, 0.12, 0.36, 0.0], [1.0, 0.72, 0.2, 0.22, 0.02]]
