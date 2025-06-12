import mujoco
import mujoco.viewer
import numpy as np
import time
import torch
from pyqtgraph.Qt import QtWidgets

from QPsolver import QP    # your QP class
from Robot.Kinematics import K_  # your kinematics helper
from Robot.Leg_Trajectory import Traj_
from Robot.Controller import C_
from Robot.Integrate import I_
from Robot.Body import B_

from util.plot import Plotter
from util.get_plot import init_plots, update_plots

# Load model & data
model = mujoco.MjModel.from_xml_path("xml/3D_Quad.xml")
data = mujoco.MjData(model)
dt = model.opt.timestep



############################################################
######################### Plotting #########################
############################################################

# initialize plotting windows
# app, ctrl_plot, state_plot, traj_plot = init_plots(data)

############################################################
######################### Mujoco #########################
############################################################

with mujoco.viewer.launch_passive(model, data) as viewer:

    # qpsolver = QP(nx=6, nu=4, data=data)
    K        = K_(data=data)
    Traj     = Traj_(data=data)
    C        = C_(data=data)
    I        = I_(data=data)
    B        = B_(data=data)

    while viewer.is_running():
        t0 = time.time()

    # Initial Pos at t=0
        if data.time < 1e-8:
            data.qpos[:3] = [0.0, 0.0, 0.3536]  
            data.qpos[5:7] = [0, 0]
            data.qpos[7] = 0 # FLHAA
            data.qpos[8] = np.pi/4# FLHIP
            data.qpos[9] = np.pi/2 # FLKNEE
            data.qpos[10] = 0. # FRHAA
            data.qpos[11] = np.pi/4 # FRHIP
            data.qpos[12] = np.pi/2. # FRKNEE
            data.qpos[13] = 0. # RLHAA
            data.qpos[14] = np.pi/4 # RLHIP
            data.qpos[15] = np.pi/2. # RLKNEE
            data.qpos[16] = 0. # RRHAA
            data.qpos[17] = np.pi/4 # RRHIP
            data.qpos[18] = np.pi/2. # RRKNEE


            # data.qpos[3:3+] = [np.pi/4, np.pi/2, 0.0, np.pi/4, np.pi/2]

        I.Cal_kinematics()

        I.Ctrl()


        ctrl_input = I.get_ctrl_input()

        # print(ctrl_input)
        data.ctrl[0:3] = ctrl_input[0]        
        data.ctrl[3:6] = ctrl_input[1]
        data.ctrl[6:9] = ctrl_input[2]
        data.ctrl[9:12] = ctrl_input[3]
        
    

            

        # if Traj.traj_mode == 0:
        #     F_front_FB = torch.zeros(2, dtype=torch.float32)
        #     F_rear_FB  = torch.zeros(2, dtype=torch.float32)
        # else:
        #     FB_input = C.FB_ctrl()
        #     F_front_FB = FB_input['F_input'].float()
        #     F_rear_FB = FB_input['R_input'].float()
        

        # FJ, RJ = torch.from_numpy(get_K['F_Jacb']).float(), torch.from_numpy(get_K['R_Jacb']).float()

        # tau_f = FJ.T @ (F_front_opt + F_front_FB)
        # tau_r = RJ.T @ (F_rear_opt + F_rear_FB)
    
        # tau_f = FJ.T @ F_front_opt
        # tau_r = RJ.T @ F_rear_opt
        
    #     # write controls
    #     data.ctrl[0:2] = tau_f.numpy()
    #     data.ctrl[2:]  = tau_r.numpy()
        # print(tau_f)

        # step

        mujoco.mj_step(model, data)





############################################################
######################### Plotting #########################
############################################################


    # update all plots
        # update_plots(data, ctrl_plot, state_plot, traj_plot, K, Traj)


############################################################
######################### Time sink ########################
############################################################
        # sync real time
        elapsed = time.time() - t0
        if dt > elapsed:
            time.sleep(dt - elapsed)
        viewer.sync()
