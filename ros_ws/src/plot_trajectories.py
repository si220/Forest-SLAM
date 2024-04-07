import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# load the pose data from the file
pose_file = 'Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P001/pose_left.txt'
data = np.loadtxt(pose_file)

# extract the position data
positions_ned = data[:, :3]

# subtract the initial position from all positions to make them start from the origin
positions_ned -= positions_ned[0]

# convert NED coordinate frame to ENU
positions_enu = np.zeros_like(positions_ned)
positions_enu[:, 0] = positions_ned[:, 1]  # East (E) from North (N)
positions_enu[:, 1] = positions_ned[:, 0]  # North (N) from East (E)
positions_enu[:, 2] = -positions_ned[:, 2]  # Up (U) from Down (D)

# create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot the trajectory in ENU frame
ax.plot(positions_enu[:, 0], positions_enu[:, 1], positions_enu[:, 2], marker='o')

# set labels for axes
ax.set_xlabel('East (E)')
ax.set_ylabel('North (N)')
ax.set_zlabel('Up (U)')

# show the plot
plt.show()
