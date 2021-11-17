import torch
from NetworkModelFile import IKNet
from robotics_src import Kinematics as kin
import numpy as np
import matplotlib.pyplot as plt
from TimingAnalysis.TimingAnalysis import TimingAnalysis as TA
timer = TA()

n = 2
dh = [[0, 0, 1, 0]] * n
rev = ['r'] * n
arm = kin.SerialArm(dh, rev)

model = IKNet()
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

for p in model.parameters():
    print(p)

qs = torch.rand((2000, 2), dtype=torch.float32) * (2*np.pi) - 1*np.pi
# qs[:, 1] = qs[:, 0] * -2
q0 = np.asarray(qs)

pos = np.zeros_like(q0)

timer.time_in("trad")
for i in range(len(q0)):
    pos[i, :] = arm.fk(q0[i, :])[0:2, 3].T
timer.time_out("trad")

y = torch.as_tensor(pos, dtype=torch.float32)
x = qs

timer.time_in("nn")
with torch.no_grad():
    yhat = model.forward(x)
timer.time_out("nn")

print(f"Net Answer: \n{yhat}\n\nTrue Answer: \n{y}")

fig, ax = plt.subplots()

for i in range(len(yhat)):
    ax.plot((yhat[i, 0], y[i, 0]), (yhat[i, 1], y[i, 1]), ls='--', color='b')

ax.scatter(yhat[:, 0], yhat[:, 1], c='r', s=20)
ax.scatter(y[:, 0], y[:, 1], c='k', s=20)

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])

timer.report_all()

plt.show()
