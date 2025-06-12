import numpy as np
import torch

class B_:

    def __init__(self, data):

        self.d = data
        self.body_pos = np.array([0,0,0,0,0,0])
        self.body_vel = np.array([0,0,0,0,0,0])

    def set_body_traj(self, t):
        """
        body PD 궤적: [beta, x, z]와 [beta_dot, x_dot, z_dot]를 반환
        """
        # t = self.d.time
        
        state_des = np.array([0,  # Roll  
                            0,  # Pitch
                            0,  # Yaw
                            0,  # x
                            0,  # y
                            0.3536,  # z
                            0,  # Roll vel 
                            0,  # Pitch vel
                            0,  # yaw vel
                            0,  # x vel
                            0,  # y vel
                            0]) # z vel
        
 
        return state_des 
        
    def sensor_data(self):
        
        d = self.d

    # Orientation (Pos)
        body_quat = d.qpos[3:7] 
        Body_orientation = self.quaternion_to_rpy(body_quat)

        lin_vel = d.qvel[0:3]     # world‐frame linear velocity
        ang_vel = d.qvel[3:6]     # world‐frame angular velocity
        
    # State

        # self.state = np.array([Body_orientation[0],  # roll
        #                      Body_orientation[1],  # pitch
        #                      Body_orientation[2],  # yaw
        #                      d.qpos[0],  # x 
        #                      d.qpos[1],  # y  
        #                      d.qpos[2],  # z
        #                      d.sensordata[3],  # roll vel  
        #                      d.sensordata[4],  # pitch vel
        #                      d.sensordata[5],  # yaw vel
        #                      d.sensordata[6],  # x vel
        #                      d.sensordata[7],  # y vel
        #                      d.sensordata[8]]) # z vel

        self.state = np.array([Body_orientation[0],  # roll
                        Body_orientation[1],  # pitch
                        Body_orientation[2],  # yaw
                        d.qpos[0],  # x 
                        d.qpos[1],  # y  
                        d.qpos[2],  # z
                        ang_vel[0],  # roll vel  
                        ang_vel[1],  # pitch vel
                        ang_vel[2],  # yaw vel
                        lin_vel[0],  # x vel
                        lin_vel[1],  # y vel
                        lin_vel[2]]) # z vel


        return self.state

    def quaternion_to_rpy(self, quat):
        """
        Quaternion to Roll-Pitch-Yaw conversion.
        Input:
        quat: array-like of length 4, in order [w, x, y, z]
        Returns:
        roll, pitch, yaw angles in radians
        """
        w, x, y, z = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        # Clamp value to avoid NaNs from out-of-range due to numerical errors
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw