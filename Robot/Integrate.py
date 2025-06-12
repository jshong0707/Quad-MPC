import numpy as np
import torch

from Robot.Kinematics import K_
from Robot.Leg_Trajectory import Traj_
from Robot.Controller import C_
from Optimization.MPC import MPCController

class I_:

    def __init__(self, data):
        
        self.d = data
        self.K = K_(data=data)
        self.Traj = Traj_(data=data)
        self.C = C_(data=data)
        self.J_input = torch.zeros(4,3)
        self.M = MPCController(data=data)

        # Optimization
        self.qp_interval = 0.03
        self.last_qp_time = -self.qp_interval


    def Cal_kinematics(self):
        
        self.get_K = self.K.Cal_Kinematics()
        get_pos_err = self.K.get_pos_err()
        
        self.pos_err_FL = get_pos_err['pos_err_FL']
        self.pos_err_FR = get_pos_err['pos_err_FR']
        self.pos_err_RL = get_pos_err['pos_err_RL']
        self.pos_err_RR = get_pos_err['pos_err_RR']
        

    def Ctrl(self):
        
        error = np.array([self.pos_err_FL, self.pos_err_FR, self.pos_err_RL, self.pos_err_RR]).flatten()
        
        

        self.FB_input = self.C.FB_ctrl(error)
        
        
        # optimization
        # run QP at interval
        if self.d.time - self.last_qp_time >= self.qp_interval:
            
            u = self.M.solve()[0]
            
            self.last_qp_time = self.d.time
            # map forces → torques
            Force_FL = torch.tensor([-u[0], -u[1], u[2]])
            Force_FR = torch.tensor([-u[3], -u[4], u[5]])
            Force_RL = torch.tensor([-u[6], -u[7], u[8]])
            Force_RR = torch.tensor([-u[9], -u[10], u[11]])

            self.opt_Force = Force_FL, Force_FR, Force_RL, Force_RR
            # print(self.opt_Force)
    def get_ctrl_input(self):
        
        J_FL = self.get_K['Jacb_FL']
        J_FR = self.get_K['Jacb_FR']
        J_RL = self.get_K['Jacb_RL']
        J_RR = self.get_K['Jacb_RR']

        # self.J_input[0] = J_FL.T @ (self.FB_input['FL_input'] + self.opt_Force[0])
        # self.J_input[1] = J_FR.T @ (self.FB_input['FR_input'] + self.opt_Force[1])
        # self.J_input[2] = J_RL.T @ (self.FB_input['RL_input'] + self.opt_Force[2])
        # self.J_input[3] = J_RR.T @ (self.FB_input['RR_input'] + self.opt_Force[3])
    
        # self.J_input[0] = J_FL.T @ (self.FB_input['FL_input'])
        # self.J_input[1] = J_FR.T @ (self.FB_input['FR_input'])
        # self.J_input[2] = J_RL.T @ (self.FB_input['RL_input'])
        # self.J_input[3] = J_RR.T @ (self.FB_input['RR_input'])

        
        self.J_input[0] = J_FL.T @ self.M.R_yaw.T @ self.opt_Force[0]
        self.J_input[1] = J_FR.T @ self.M.R_yaw.T @ self.opt_Force[1]
        self.J_input[2] = J_RL.T @ self.M.R_yaw.T @ self.opt_Force[2]
        self.J_input[3] = J_RR.T @ self.M.R_yaw.T @ self.opt_Force[3]
        
        # print(J_FL.T , self.opt_Force[0])
        




        # print(self.opt_Force[0][2], self.opt_Force[1][2], self.opt_Force[2][2], self.opt_Force[3][2])
        
        return np.array([self.J_input[0],
                         self.J_input[1],
                         self.J_input[2],
                         self.J_input[3]])
    