from __future__ import annotations

import numpy as np

from .math_utils import quaternion_to_R, rot_to_rpy_zxy, wrap_to_pi
from .model import QuadParams


class Controller:
    """Controller for quadrotor trajectory tracking."""

    def __init__(self, params: QuadParams, mode: str = "model_based") -> None:
        """Initialize controller with quadrotor parameters."""
        self.params = params
        self.mode = mode

        # ========================================================================
        # 기본 게인 값 (튜닝)
        # ========================================================================
        self.Kp_pos = np.array([10.0, 10.0, 15.0])
        self.Kd_pos = np.array([5.0, 5.0, 8.0])
        
        self.Kp_angle = np.array([100.0, 100.0, 100.0])
        self.Kd_angle = np.array([10.0, 10.0, 10.0])

    def reset(self) -> None:
        """Reset controller state"""
        pass

    def __call__(self, t: float, s: np.ndarray, s_des: np.ndarray) -> tuple[float, np.ndarray]:
        m = self.params.mass
        g = self.params.grav
        I = self.params.I

        # ========================================================================
        # 1. Parse Current & Desired States
        # ========================================================================
        pos = s[0:3]
        vel = s[3:6]
        quat = s[6:10]
        omega = s[10:13] # Body angular velocities [p, q, r]

        pos_des = s_des[0:3]
        vel_des = s_des[3:6]
        acc_des = s_des[6:9]
        psi_des = s_des[9]
        psi_dot_des = s_des[10]

        R = quaternion_to_R(quat)
        phi, theta, psi = rot_to_rpy_zxy(R)

        # ========================================================================
        # 2. Position Controller
        # ========================================================================
        e_pos = pos_des - pos
        e_vel = vel_des - vel
        
        # Feedforward / Kp-only 모드 구분
        if self.mode == "no_forward":
            acc_c = self.Kp_pos * e_pos + self.Kd_pos * e_vel
        elif self.mode == "no_forward_kp_only":
            acc_c = self.Kp_pos * e_pos
        else:
            acc_c = acc_des + self.Kp_pos * e_pos + self.Kd_pos * e_vel

        # model based라서 기본적으로 줘야하는 중력 들어있음
        F = m * (g + acc_c[2]) 

        # ========================================================================
        # 3. Attitude Command Calculation
        # ========================================================================
        # 가속도 명령을 위해 달성해야하는 롤피치
        phi_c = (1.0 / g) * (acc_c[0] * np.sin(psi) - acc_c[1] * np.cos(psi))
        theta_c = (1.0 / g) * (acc_c[0] * np.cos(psi) + acc_c[1] * np.sin(psi))

        # ========================================================================
        # 4. Attitude Controller
        # ========================================================================
        e_phi = wrap_to_pi(phi_c - phi)
        e_theta = wrap_to_pi(theta_c - theta)
        e_psi = wrap_to_pi(psi_des - psi)
        
        ang_pos_err = np.array([e_phi, e_theta, e_psi])

        # Feedforward의 유무 모드 구분
        if self.mode == "no_forward":
            ang_rate_err = np.array([0.0, 0.0, 0.0]) - omega
        else:
            ang_rate_err = np.array([0.0, 0.0, psi_dot_des]) - omega

        alpha_c = self.Kp_angle * ang_pos_err + self.Kd_angle * ang_rate_err

        # 관성 모멘트 보상 역시 모드에 상관없이 항상 해줍니다.
        M = I @ alpha_c + np.cross(omega, I @ omega)

        return F, M