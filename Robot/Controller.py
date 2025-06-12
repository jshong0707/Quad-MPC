import numpy as np
import torch
from Robot.Kinematics import K_
from Robot.Leg_Trajectory import Traj_

class C_:
    def __init__(self, data, cutoff_freq=70, Ts=0.001):
        """
        kine: Kinematics instance for Jacobian etc.
        PID gains explicitly defined for each of 12 DoFs.
        cutoff_freq: low-pass filter cutoff frequency for derivative term
        Ts: sampling time
        """
        self.Traj = Traj_(data=data)

        # Explicit gains for each DoF
        self.KP = np.array([
            8000, 10000, 8000, 
            8000, 10000, 7000,
            8000, 10000, 6000,
            8000, 10000, 6000
        ], dtype=float)

        self.KI = np.array([
            100, 100, 100,
            100, 100, 100,
            100, 100, 100,
            100, 100, 100
        ], dtype=float)

        self.KD = np.array([
            200, 200, 200, 
            200, 180, 180,
            180, 180, 160,
            160, 160, 160
        ], dtype=float)

        # discretization
        self.cutoff_freq = float(cutoff_freq)
        self.Ts = float(Ts)
        # compute tau for derivative filter
        self.tau = 1.0 / (2.0 * np.pi * self.cutoff_freq)
        # internal states
        self.error_old = np.zeros_like(self.KP)
        self.I_term_old = np.zeros_like(self.KI)
        self.D_term_old = np.zeros_like(self.KD)

    def pid(self, error: np.ndarray) -> np.ndarray:
        """
        Compute PID output using Tustin (bilinear) discretization:
          P = KP * error
          I = I_old + KI * (Ts/2 * (error + error_old))
          D = (2*KD/(2*tau+Ts))*(error-error_old) - ((Ts-2*tau)/(2*tau+Ts))*D_old
        error_old, I_term_old, D_term_old updated internally.
        """

        # proportional term
        P_term = self.KP * error
        # integral term (trapezoidal rule)
        I_term = self.I_term_old + self.KI * (self.Ts / 2.0) * (error + self.error_old)
        # derivative term (Tustin discretization)
        D_term = (2.0 * self.KD / (2.0 * self.tau + self.Ts)) * (error - self.error_old)
        D_term -= ((self.Ts - 2.0 * self.tau) / (2.0 * self.tau + self.Ts)) * self.D_term_old
        # PID output
        output = P_term + I_term + D_term
        # update internal states
        self.error_old = error.copy()
        self.I_term_old = I_term.copy()
        self.D_term_old = D_term.copy()
        return output

    def update(self):
        # placeholder if you want to integrate kinematics-based feedforward
        pass

    def FB_ctrl(self, error):

        ctrl_input = self.pid(error)
        FL_input = torch.tensor(ctrl_input[0:3])
        FR_input = torch.tensor(ctrl_input[3:6])
        RL_input = torch.tensor(ctrl_input[6:9])
        RR_input = torch.tensor(ctrl_input[9:12])

        return {
            'FL_input': FL_input,
            'FR_input': FR_input,
            'RL_input': RL_input,
            'RR_input': RR_input
        }
