import pybullet as p
import pybullet_data
import math
import pybullet_planning as pbp
import time


p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)

boxId = p.loadURDF("plane.urdf")
fr5 = p.loadURDF("F:\\Pycharm_project\\RL\\fr5_description\\urdf\\fr5_robot.urdf",useFixedBase=True)
target = p.loadURDF("F:\\Pycharm_project\\RL\\fr5_description\\urdf\\box.urdf",
                    [0.5, 0.0, 0.0],
                    p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=False,
                    globalScaling=0.4,
                    flags=p.URDF_USE_SELF_COLLISION)

p.setJointMotorControlArray(fr5,range(7),p.POSITION_CONTROL,targetPositions=[0.0,0.0,-math.pi/2,0.0,-math.pi/2,0.0,0.0])


while 1:
     p.stepSimulation()
     time.sleep(1./240.)     