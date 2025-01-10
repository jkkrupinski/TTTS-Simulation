from xarm.wrapper import XArmAPI


# Connect to the simulated robot
arm = XArmAPI('192.168.1.238')  # Replace with the IP address used in uFactory Studio

arm.set_state(0)  

arm.set_simulation_robot(False)


arm.set_position(x=320, y=40, z=200, roll=-180, pitch=0, yaw=-15, speed=20, wait=True)
arm.set_position(x=320, y=0, z=200, roll=-180, pitch=0, yaw=-15, speed=20, wait=True)
arm.set_position(x=280, y=0, z=200, roll=-180, pitch=0, yaw=-15, speed=20, wait=True)
arm.set_position(x=280, y=40, z=200, roll=-180, pitch=0, yaw=-15, speed=20, wait=True)


# Disconnect after the operation
arm.disconnect()