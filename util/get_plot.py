# get_plot.py
import numpy as np
from pyqtgraph.Qt import QtWidgets
from util.plot import Plotter

def init_plots(data):
    """
    Create three Plotter windows:
     1) Ctrl inputs (5 subplots, 1 curve each)
     2) Body states (6 subplots, 2 curves each)
     3) Leg trajectories (4 subplots, 2 curves each)
    Returns (app, ctrl_plot, state_plot, traj_plot)
    """
    # ensure a single QApplication
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    # 1) Ctrl inputs
    ctrl_titles = ["Front Hip τ", "Front Knee τ", "Spine τ", "Rear Hip τ", "Rear Knee τ"]
    ctrl_ylims   = [(-50,50)] * 5
    ctrl_plot = Plotter(
        data,
        titles=ctrl_titles,
        Big_title="Ctrl Inputs",
        n=5,
        interval=0.02,
        ylims=ctrl_ylims,
        num_curves=1,
        pens=['y']
    )
    ctrl_plot.win.show()
    ctrl_plot.win.move(1700, 100)

    # 2) Body states
    state_titles = ["Pitch","x","z","Pitch d","xd","zd"]
    state_ylims   = [(-0.2,0.2),(-0.2,0.2),(0,0.5),(-0.2,0.2),(-0.2,0.2),(-0.2,0.2)]
    state_plot = Plotter(
        data,
        titles=state_titles,
        Big_title="Body States",
        n=6,
        interval=0.02,
        ylims=state_ylims,
        num_curves=2,
        pens=['y','w']
    )
    state_plot.win.show()
    state_plot.win.move(2570, 100)

    # 3) Leg trajectories
    traj_titles = ["Front x","Front z","Rear x","Rear z"]
    traj_ylims   = [(-0.5,0.5),(0,0.5),(-0.5,0.5),(0,0.5)]
    traj_plot = Plotter(
        data,
        titles=traj_titles,
        Big_title="Leg Trajectories",
        n=4,
        interval=0.02,
        ylims=traj_ylims,
        num_curves=2,
        pens=['y','w']
    )
    traj_plot.win.show()
    traj_plot.win.move(3200, 100)

    return app, ctrl_plot, state_plot, traj_plot


def update_plots(data, ctrl_plot, state_plot, traj_plot, K, Traj):
    """
    Push new data into each Plotter and redraw if interval elapsed.
    K   : Kinematics instance
    Traj: Trajectory instance
    """
    # 1) Control inputs
    ctrl_vals = [data.ctrl[i] for i in range(5)]
    ctrl_plot.push(ctrl_vals)
    ctrl_plot.update()

    # 2) Body states: actual vs desired
    actual = [
        data.qpos[2], data.qpos[0], data.qpos[1],
        data.qvel[2], data.qvel[0], data.qvel[1],
    ]
    bt = Traj.set_body_traj()
    desired = list(bt['pos_des']) + list(bt['vel_des'])
    state_plot.push(list(zip(actual, desired)))
    state_plot.update()

    # 3) Leg trajectories: actual vs desired
    kin = K.Cal_Kinematics()
    actual_leg = [
        kin['F_pos'][0], kin['F_pos'][1],
        kin['R_pos'][0], kin['R_pos'][1]
    ]
    dp = Traj.get_des_pos()
    desired_leg = [
        dp['F_pos_des'][0], dp['F_pos_des'][1],
        dp['R_pos_des'][0], dp['R_pos_des'][1]
    ]
    traj_plot.push(list(zip(actual_leg, desired_leg)))
    traj_plot.update()

    # pump Qt events to keep GUI responsive
    QtWidgets.QApplication.processEvents()
