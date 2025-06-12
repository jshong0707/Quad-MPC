# Robot/Trajectory.py
import numpy as np
# from Robot.Kinematics import K_

# Stance leg는 desired position이 현재 position과 같게 나와서 error = zero

class Traj_:
    def __init__(self, data):
        # MuJoCo data handle
        self.d = data
        # Kinematics helper
        # self.K = K_(data=data)
        
        # Trajectory
        self.traj_mode = 0 # 0: Only FB , 1: QP(Stance),  2: Trotting  

        # timing parameters (seconds)
        self.stance_time = 0.2
        self.swing_time  = 0.1
        self.period      = self.stance_time + self.swing_time

        # --- Front leg bookkeeping ---
        self.F_swing_phase = False
        self.F_start       = None   # swing 시작 시점의 front foot pos

        # --- Rear leg bookkeeping ---
        self.R_swing_phase = False
        self.R_start       = None   # swing 시작 시점의 rear  foot pos

        self.z0 = 0.3536

        # swing 목표 위치 (x,z)
        self.F_target = np.array([0.0, self.z0])
        self.R_target = np.array([0.0, self.z0])

        # 얼마나 띄워 올릴지 (m)
        self.lift_height = -0.2

        # initial position
        self.F_pos_des = np.array([0, self.z0])
        self.R_pos_des = np.array([0, self.z0])
        

    def custom_leg_pos_traj(self):
        
        t = self.d.time

        pos_des_FL = np.array([0. + 0.1*np.sin(t),
                               0 ,
                               0.3536]) 
        pos_des_FR = np.array([0 + 0.1*np.sin(t),
                               0,
                               0.3536]) 
        pos_des_RL = np.array([0 + 0.1*np.sin(t),
                               0,
                               0.3536]) 
        pos_des_RR = np.array([0 + 0.1*np.sin(t),
                               0,
                               0.3536]) 

        return {'pos_des_FL': pos_des_FL,
                'pos_des_FR': pos_des_FR,
                'pos_des_RL': pos_des_RL,
                'pos_des_RR': pos_des_RR}
    
    def set_swing_leg_traj(self):
        """
        Front/Rear 각각
         - stance: 실제 위치 반환
         - swing: Bézier 보간된 위치 반환
        """
        t = self.d.time
        # 1) 앞다리 위상
        t_mod_F = t % self.period
        # 2) 뒷다리 위상 (앞다리보다 stance_time만큼 offset)
        t_mod_R = (t_mod_F + self.stance_time) % self.period

        # 매 호출마다 실제 위치 업데이트
        self._update_current_pos()

        # --- Front leg trajectory ---
        if t_mod_F < self.stance_time:
            # Front stance
            self.F_swing_phase = False
            F_out = self.Fpos.copy()
        else:
            # Front swing
            tau_F = (t_mod_F - self.stance_time) / self.swing_time  # [0,1]
            if not self.F_swing_phase:
                # swing 시작 시 한 번만 초기 위치 저장
                self.F_swing_phase = True
                self.F_start = self.Fpos.copy()
            F_out = self._bezier_interp(self.F_start, self.F_target, tau_F)

        # --- Rear leg trajectory ---
        if t_mod_R < self.stance_time:
            # Rear stance
            self.R_swing_phase = False
            R_out = self.Rpos.copy()
        
        else:
            # Rear swing
            tau_R = (t_mod_R - self.stance_time) / self.swing_time
            if not self.R_swing_phase:
                self.R_swing_phase = True
                self.R_start = self.Rpos.copy()
            R_out = self._bezier_interp(self.R_start, self.R_target, tau_R)

        return {'F_pos': np.array(F_out), 'R_pos': np.array(R_out)}

    def _bezier_interp(self, P0, P3, tau):
        """
        Cubic Bézier 보간:
         P0→P3, 중간에 peak 높이만큼 띄워줌
        """
        # control points
        peak = np.array([
            0.5*(P0[0] + P3[0]),
            max(P0[1], P3[1]) + self.lift_height
        ])
        P1 = P0 + 0.25*(peak - P0)
        P2 = P3 + 0.25*(peak - P3)
        # Bézier 공식
        return (
            (1 - tau)**3 * P0 +
            3*(1 - tau)**2 * tau * P1 +
            3*(1 - tau)  * tau**2 * P2 +
            tau**3 * P3
        )

    def get_des_pos(self):
    
        if self.traj_mode == 0:
            traj = self.custom_leg_pos_traj()
        elif self.traj_mode == 2: 
            traj = self.set_swing_leg_traj()
        else:
            traj = self.pos_Hold()

        self.pos_des_FL = traj['pos_des_FL']
        self.pos_des_FR = traj['pos_des_FR']
        self.pos_des_RL = traj['pos_des_RL']
        self.pos_des_RR = traj['pos_des_RR']
        
        return {'pos_des_FL': self.pos_des_FL,
                'pos_des_FR': self.pos_des_FR,
                'pos_des_RL': self.pos_des_RL,
                'pos_des_RR': self.pos_des_RR}
                

    def pos_Hold(self):
        
        pos_des_FL = np.array([0, 0, 0.3536]) 
        pos_des_FR = np.array([0, 0, 0.3536]) 
        pos_des_RL = np.array([0, 0, 0.3536]) 
        pos_des_RR = np.array([0, 0, 0.3536])

        return {'pos_des_FL': pos_des_FL,
                'pos_des_FR': pos_des_FR,
                'pos_des_RL': pos_des_RL,
                'pos_des_RR': pos_des_RR}

    # def _update_current_pos(self):
    #     """매 호출 시점마다 Kinematics로부터 실제 발 위치를 받아 저장"""
    #     kin = self.K.Cal_Kinematics()
    #     self.Fpos = kin['F_pos']  # shape (2,) = [x, z]
    #     self.Rpos = kin['R_pos']

    def Cal_error(self):
        """
        현재 desired swing trajectory 대비 실제 위치 오차 계산
        → {'F_err': Δfront, 'R_err': Δrear}
        """
        # desired 위치
        traj = self.get_des_pos()
        
        self.pos_des_FL = traj['pos_des_FL']
        self.pos_des_FR = traj['pos_des_FR']
        self.pos_des_RL = traj['pos_des_RL']
        self.pos_des_RR = traj['pos_des_RR']

        pos_err_FL = self.pos_des_FL - self.pos_FL
        
        # 실제 위치 다시 업데이트
        self._update_current_pos()

        F_pos_err = self.F_pos_des - self.Fpos
        R_pos_err = self.R_pos_des - self.Rpos
        return {'F_pos_err': np.array(F_pos_err), 'R_pos_err': np.array(R_pos_err)}

