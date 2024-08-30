import ikpy.chain
import numpy as np

import ikpy.utils.plot as plot_utils

active_links_mask = [
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
]  # Exclude base_link and the end-effector and tool

my_chain = ikpy.chain.Chain.from_urdf_file(
    active_links_mask=active_links_mask, urdf_file="scene/ufactory_xarm7/adad.urdf"
)


target_position = [0.4, 0.6, 0.4]
# print(my_chain.links)
# my_chain.forward_kinematics([0, 0, 0, 0, 0, 0, 0])


real_frame = my_chain.forward_kinematics(my_chain.inverse_kinematics(target_position))
print(
    "Computed position vector : %s, original position vector : %s"
    % (real_frame[:3, 3], target_position)
)

orientation_axis = "Y"
target_orientation = [0, 0, 1]

import matplotlib.pyplot as plt

fig, ax = plot_utils.init_3d_figure()

joints = my_chain.inverse_kinematics(
    target_position,
    # target_orientation=target_orientation,
    # orientation_mode=orientation_axis,
)

joints = [0, 0, -.247, 0, .909, 0, 1.15644, 0]


my_chain.plot(
    joints,
    ax,
    target=target_position,
)
print(joints)


# my_chain.plot(
#     my_chain.inverse_kinematics(target_position, orientation_mode=None),
#     ax,
#     target=target_position,
# )
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)

plt.show()

