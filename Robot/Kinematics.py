import numpy as np
import torch
from Robot.Leg_Trajectory import Traj_

class K_:

    def __init__(self, data):
        
        self.Traj = Traj_(data=data)
        self.d = data
        self.L = 0.25  # Link length
        self.L_hip = 0.0

        
    def get_joint_ang(self, num):
        
        base = 3*num + 7
        base_vel = 3*num + 6
        q = self.d.qpos[base : base + 3]
        qd = self.d.qvel[base_vel : base_vel + 3]
        
        return q, qd
    
    def get_foot_pos(self, q):
        L, L_hip = self.L, self.L_hip
        q0, q1, q2 = q

        x = -L * (np.cos(q1) + np.cos(q1 + q2))
        y = L_hip * np.cos(q0) + L*(np.sin(q1) + np.sin(q1 + q2))*np.sin(q0)
        z = L * (np.sin(q1) + np.sin(q1 + q2)) * np.cos(q0) - L_hip * np.sin(q0)

        # 합산
        return np.array([x, y, z], dtype=float)

    def get_Jacb(self, q):

        L     = self.L
        L_hip = self.L_hip
        q0, q1, q2 = q


        j11 = 0.0
        j12 = L * (np.sin(q1) + np.sin(q1 + q2))
        j13 = L * np.sin(q1 + q2)

        j21 = -L_hip * np.sin(q0) + L * np.cos(q0) * (np.sin(q1) + np.sin(q1+q2))
        j22 = L * (np.cos(q1) + np.cos(q1+q2)) * np.sin(q0)
        j23 = L * np.cos(q1 + q2) * np.sin(q0)

        j31 = -L_hip * np.cos(q0) - L * np.sin(q0) * (np.sin(q1) + np.sin(q1+q2))
        j32 = L * (np.cos(q1 + q2) + np.cos(q1)) * np.cos(q0)
        j33 = L * np.cos(q1 + q2) * np.cos(q0)

        J = torch.tensor(np.array([
            [j11, j12, j13],
            [j21, j22, j23],
            [j31, j32, j33]
        ], dtype=float))

        return J
    
    def Cal_Kinematics(self):
        
        q_FL, qd_FL = self.get_joint_ang(0)
        q_FR, qd_FR = self.get_joint_ang(1)
        q_RL, qd_RL = self.get_joint_ang(2)
        q_RR, qd_RR = self.get_joint_ang(3)

        self.pos_FL = self.get_foot_pos(q=q_FL)
        self.pos_FR = self.get_foot_pos(q=q_FR)
        self.pos_RL = self.get_foot_pos(q=q_RL)
        self.pos_RR = self.get_foot_pos(q=q_RR)


        J_FL = self.get_Jacb(q_FL)
        J_FR = self.get_Jacb(q_FR)
        J_RL = self.get_Jacb(q_RL)
        J_RR = self.get_Jacb(q_RR)

        self.vel_FL = J_FL @ qd_FL
        self.vel_FR = J_FR @ qd_FR
        self.vel_RL = J_RL @ qd_RL
        self.vel_RR = J_RR @ qd_RR
        
    


        return {
            'pos_FL':   self.pos_FL,
            'pos_FR':   self.pos_FR,
            'pos_RL':   self.pos_RL,
            'pos_RR':   self.pos_RR,
            'Jacb_FL':  J_FL,
            'Jacb_FR':  J_FR,
            'Jacb_RL':  J_RL,
            'Jacb_RR':  J_RR,
            'vel_FL':   self.vel_FL,
            'vel_FR':   self.vel_FR,
            'vel_RL':   self.vel_RL,
            'vel_RR':   self.vel_RR
        }

    def get_pos_err(self):
        traj = self.Traj.get_des_pos()

        pos_err_FL = traj['pos_des_FL'] - self.pos_FL
        pos_err_FR = traj['pos_des_FR'] - self.pos_FR
        pos_err_RL = traj['pos_des_RL'] - self.pos_RL
        pos_err_RR = traj['pos_des_RR'] - self.pos_RR

        return {'pos_err_FL': pos_err_FL,
                'pos_err_FR': pos_err_FR,
                'pos_err_RL': pos_err_RL,
                'pos_err_RR': pos_err_RR}