import casadi as ca
import numpy as np
from scipy.linalg import expm
from Robot.Body import B_
from Robot.Kinematics import K_

class MPCController:
    def __init__(self, data, horizon=10, dt=0.03):
        """
        data    : mujoco data handle
        horizon : 예측 지평선 길이 (스텝 수)
        dt      : 샘플링 타임
        """
        # interfaces
        self.Body = B_(data=data)
        self.K    = K_(data=data)
        self.data = data
        self.N    = horizon
        self.dt   = dt

        # Robot parameters (constant)
        self.M      = 30.8
        self.I_B            = np.diag([0.0674, 0.41734, 0.48334])*2

        # Friction coefficient
        self.mu = 0.6
        
        # 임의 초기 분산화 (실제 값은 solve() 에서 덮어씀)
        Ad0, Bd0, gd0 = self._discretize_dynamics(self.dt)
        self.nx = Ad0.shape[0]
        self.nu = Bd0.shape[1]

        
        # Cost weights
        self.Q  = np.diag([1000.,10.,10., 1000.,1000.,100., 100.,1.,1., 1.,1.,1.]) 
        # self.Q  = np.diag([0.,0,0., 0.,0.,0, 0.,0.,0., 0.,0.,0.])
        
        self.R  = np.diag([1 ,1 ,1, 1.,1.,1, 1.,1.,1, 1.,1.,1.]) * 0.01
        
        self.Qf = self.Q * 100.0

        # Build CasADi Opti problem once
        opti = ca.Opti()

        # 1) only U is an optimization variable
        self.U       = opti.variable(self.nu, self.N)

        # 2) parameters: initial state, reference traj, and dynamics matrices
        self.X0      = opti.parameter(self.nx)
        self.X_des   = opti.parameter(self.nx, self.N+1)
        self.Ad_p    = opti.parameter(self.nx, self.nx)
        self.Bd_p    = opti.parameter(self.nx, self.nu)
        self.gd_p    = opti.parameter(self.nx)

        # 3) build state trajectory from U via *parameterized* dynamics
        Xk = self.X0
        X_seq = [Xk]
        for k in range(self.N):
            Xk = self.Ad_p @ Xk + self.Bd_p @ self.U[:,k] + self.gd_p
            X_seq.append(Xk)

        # 4) objective
        cost = 0
        for k in range(self.N):
            dx = X_seq[k]   - self.X_des[:,k]
            uk = self.U[:,k]
            cost += dx.T @ self.Q @ dx + uk.T @ self.R @ uk
        dN = X_seq[self.N] - self.X_des[:,self.N]
        

        cost += dN.T @ self.Qf @ dN
        opti.minimize(cost)

        # 5) input bounds
        opti.subject_to(opti.bounded(-100, self.U, 200))
        # for idx in (2,5,8,11):
        #     opti.subject_to(self.U[idx,:] >= 50)

                # per-leg friction constraints
        for k in range(self.N):
            for i in range(4):
                fx = self.U[3*i + 0, k]
                fy = self.U[3*i + 1, k]
                fz = self.U[3*i + 2, k]
                # normal force ≥ 0
                opti.subject_to(fz >= 0)
                # |fx| ≤ μ fz
                opti.subject_to(fx <=  self.mu * fz)
                opti.subject_to(fx >= -self.mu * fz)
                # |fy| ≤ μ fz
                opti.subject_to(fy <=  self.mu * fz)
                opti.subject_to(fy >= -self.mu * fz)

        # 6) solver settings
        opti.solver('ipopt', {
            'print_time': False,
            'print_in':   False,
            'print_out':  False,
            'ipopt.print_level': 0
        })

        self.opti = opti

    def skew(self, v):
        return np.array([
            [    0, -v[2],  v[1]],
            [ v[2],     0, -v[0]],
            [-v[1],  v[0],     0]
        ])

    def _discretize_dynamics(self, T):
        # Continuous orientation
        body_state = self.Body.sensor_data()
        yaw = body_state[2] 

        # Leg position vectors
        com = self.data.subtree_com[0]
        r_list = [self.data.site_xpos[i] - com for i in (1,2,3,4)]

        # Continuous-time A
        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw),  np.cos(yaw), 0],
                          [0,            0,           1]])
        
        self.R_yaw = R_yaw

        Z3, I3 = np.zeros((3,3)), np.eye(3)
        A = np.block([
            [Z3, Z3,    I3, Z3],
            [Z3, Z3,    Z3,    I3],
            [Z3, Z3,    Z3,    Z3],
            [Z3, Z3,    Z3,    Z3],
        ])

        I_W = R_yaw @ self.I_B @ R_yaw.T
        I_W_inv = np.linalg.inv(I_W)

        # Continuous-time B
        B_rows = [
            [Z3, Z3, Z3, Z3],
            [Z3, Z3, Z3, Z3],
            [I_W_inv @ self.skew(r) for r in r_list],
            [I3 / self.M for _ in range(4)]
        ]

        B = np.block(B_rows)

        # Gravity vector
        g = np.zeros(12); g[9:12] = [0, 0, -9.81]

        # Zero-order hold discretization
        Ad = expm(A * T)
        integral = np.zeros_like(A)
        for τ in np.linspace(0, T, 100):
            integral += expm(A * τ) * (T/100)
        Bd = integral @ B
        gd = integral @ g

        return Ad, Bd, gd

    def solve(self):
        # 1) 매번 현재 yaw 등에 맞춰 dynamics 재이산화
        Ad, Bd, gd = self._discretize_dynamics(self.dt)


        # 2) 현재 시각 읽기
        t0 = self.data.time

        # 3) 미래 궤적 생성: [X*(t0), X*(t0+dt), ..., X*(t0+N*dt)]
        x_des = np.stack([
            self.Body.set_body_traj(t0 + k*self.dt)
            for k in range(self.N+1)
        ], axis=0)  # shape (N+1, 12)

        # 2) parameters 세팅
        x     = self.Body.sensor_data()

        self.opti.set_value(self.X0,    x)
        self.opti.set_value(self.X_des, x_des.T)
        self.opti.set_value(self.Ad_p,  Ad)
        self.opti.set_value(self.Bd_p,  Bd)
        self.opti.set_value(self.gd_p,  gd)


        
        # 3) solve
        sol = self.opti.solve()

        return sol.value(self.U).T  # shape (N, nu)
