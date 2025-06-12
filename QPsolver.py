import torch
import numpy as np
import qpth
from qpth.qp import QPFunction
from scipy.linalg import expm
from Robot.Leg_Trajectory import Traj_


class QP:
    def __init__(self, data, lb=None, ub=None):
        self.data = data

        # PDIPM solver 세팅
        self.solver = QPFunction(verbose=False, eps=1e-6, maxIter=200)

        # 힘(f) 변수만 최적화하니 nu=4
        self.nu = 4
        # inequality bound
        self.lb = torch.tensor([-300., -300., -300., -300.])
        self.ub = torch.tensor([ 300., 300.,  300., 300.])

        # 로봇 파라미터
        self.M   = 30.0
        self.Izz = 0.77
        self.g   = 9.81

        # PD gains for desired accel
        self.kp = np.diag([1000., 1000., 700.])   # [beta, x, z]
        self.kd = np.diag([200., 200., 50.])

        # cost weights
        self.S = torch.diag(torch.tensor([2., 2., 1.]))      # dynamics residual 가중치
        self.R = torch.eye(self.nu) * 5                # f^T R f 가중치

        self.Traj = Traj_(self.data)
        self.traj_mode = self.Traj.traj_mode

    def cost(self):
        # 1) 현재 상태
        beta,     x,   z     = self.data.qpos[2], self.data.qpos[0], self.data.qpos[1]
        beta_dot, x_dot, z_dot = self.data.qvel[2], self.data.qvel[0], self.data.qvel[1]

        # 2) Desired State
        des_motion = self.Traj.set_body_traj()
        beta_des, x_des, z_des     = des_motion['pos_des']
        betad_des, x_dot_des, z_dot_des = des_motion['vel_des']

        # 3) PD로 desired accel 계산
        e_pos = np.array([beta_des - beta,  x_des - x,   z_des - z])
        e_dot = np.array([betad_des - beta_dot,  x_dot_des - x_dot,  z_dot_des - z_dot])
        des_acc = self.kp.dot(e_pos) + self.kd.dot(e_dot)    # (3,)

    # 4) dynamics RHS 벡터 b
    #   [ Izz⋅betä ;  M⋅ẍ ; M⋅z̈ + M⋅g ]
        b_np = np.array([
            self.Izz * des_acc[0],
            self.M   * des_acc[1],
            self.M   * des_acc[2] + self.M * self.g
        ], dtype=np.float32)

    # 5) A 매트릭스
        Fp = self.data.site_xpos[1] - self.data.subtree_com[0]
        Rp = self.data.site_xpos[2] - self.data.subtree_com[0]
        rFz, rFx = float(Fp[2]), float(Fp[0])
        rRz, rRx = float(Rp[2]), float(Rp[0])
        A_np = np.array([
            [  rFz,  -rFx,   rRz,  -rRx],
            [    1,     0,     1,     0],
            [    0,     1,     0,     1],
        ], dtype=np.float32)

    # 6) torch 변환
        b = torch.from_numpy(b_np)       # (3,)
        A = torch.from_numpy(A_np)       # (3,4)
        S = self.S.float()               # (3,3)
        R = self.R.float()               # (4,4)

    # 7) QP H, f 계산
    #    J = (A f - b)^T S (A f - b) + f^T R f
    # 전개하면: 1/2 f^T H f + f^T f_term  (+ const)
        H = 2*( A.t() @ S @ A + R )      # (4,4)
        f =   -2*( b.t() @ S @ A )       # (4,)

    # 8) 부등식 제약 f_lb ≤ f ≤ f_ub  →  [I; -I] f ≤ [f_ub; -f_lb]


        lb = self.lb.clone()
        ub = self.ub.clone()
        
        if(self.traj_mode == 2):
            _ = self.Traj.set_swing_leg_traj()

            # print("F: ", self.Traj.F_swing_phase, "R: ", self.Traj.R_swing_phase)
            # 앞다리 스윙 중이면 force[0], force[1] = 0 으로 고정
            if self.Traj.F_swing_phase:
                lb[0:2] = 0.0
                ub[0:2] = 0.0

            # 뒷다리 스윙 중이면 force[2], force[3] = 0 으로 고정
            if self.Traj.R_swing_phase:
                lb[2:4] = 0.0
                ub[2:4] = 0.0
        
        
        G = torch.cat([torch.eye(self.nu), -torch.eye(self.nu)], dim=0)   # (8,4)
        h = torch.cat([ ub, -lb ], dim=0)                  # (8,)

            # (10) 마찰력 제약 추가 (μ는 마찰계수)
        mu = 0.5  # 예: μ = 0.5
        G_fric = torch.tensor([
            [ 1.0, -mu,  0.0,  0.0],  #  Fx - μ Fz ≤ 0
            [-1.0, -mu,  0.0,  0.0],  # -Fx - μ Fz ≤ 0
            [ 0.0,  0.0,  1.0, -mu ], #  Fx(rear) - μ Fz(rear) ≤ 0
            [ 0.0,  0.0, -1.0, -mu ], # -Fx(rear) - μ Fz(rear) ≤ 0
        ], dtype=H.dtype, device=H.device)  # (4,4)
        h_fric = torch.zeros(4, dtype=H.dtype, device=H.device)

        # (11) 합치기
        G_ineq = torch.cat([G, G_fric], dim=0)  # (12,4)
        h_ineq = torch.cat([h, h_fric], dim=0)  # (12,)

    # batch 차원
        H = H.unsqueeze(0)   # (1,4,4)
        f = f.unsqueeze(0)   # (1,4)
        G = G_ineq.unsqueeze(0)   # (1,8,4)
        h = h_ineq.unsqueeze(0)   # (1,8)
        A_eq = torch.zeros((1,0,self.nu), dtype=H.dtype, device=H.device)
        b_eq = torch.zeros((1,0),        dtype=H.dtype, device=H.device)

        return H, f, G, h, A_eq, b_eq
