from robotics_src import Kinematics as kin
import csv
import numpy as np

n = 2
p_train = 500000
p_test = 10000
dh = [[0, 0, 1, 0]] * n
joint = ['r'] * n
arm = kin.SerialArm(dh, joint)

qs_train = np.random.random_sample((p_train, n)) * 0.2 - 0.1
qs_train *= np.pi

qs_test = np.random.random_sample((p_test, n)) * 0.2 - 0.1
qs_test *= np.pi

ts_train = np.zeros_like(qs_train)
ts_test = np.zeros_like(qs_test)

print("Generating Training Data --------------")
for i, q in enumerate(qs_train):
    T = arm.fk_fast(q[0], q[1])
    ts_train[i, :] = T[0:2, 3]

    if i % 1000 == 0:
        print(f"{i} / {p_train}")

print("Generating Test Data --------------")
for i, q in enumerate(qs_test):
    T = arm.fk_fast(q[0], q[1])
    ts_test[i, :] = T[0:2, 3]

    if i % 100 == 0:
        print(f"{i} / {p_test}")

print("Saving CSV Files")

filename = 'two_link_training_data.csv'
with open(filename, mode='w') as csvfile:
    csv_writer = csv.writer(csvfile)

    for t in ts_train:
        csv_writer.writerow([t[0], t[1]])

filename = 'two_link_training_label.csv'
with open(filename, mode='w') as csvfile:
    csv_writer = csv.writer(csvfile)

    for q in qs_train:
        csv_writer.writerow([q[0], q[1]])

filename = 'two_link_test_data.csv'
with open(filename, mode='w') as csvfile:
    csv_writer = csv.writer(csvfile)

    for t in ts_test:
        csv_writer.writerow([t[0], t[1]])

filename = 'two_link_test_label.csv'
with open(filename, mode='w') as csvfile:
    csv_writer = csv.writer(csvfile)

    for q in qs_test:
        csv_writer.writerow([q[0], q[1]])

print('finished')