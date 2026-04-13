# from __future__ import annotations

# import numpy as np

# from .math_utils import quaternion_to_R, rot_to_rpy_zxy, wrap_to_pi
# from .model import QuadParams


# class Controller:
#     """Controller for quadrotor trajectory tracking."""

#     def __init__(self, params: QuadParams, mode: str = "model_based") -> None:
#         """Initialize controller with quadrotor parameters."""
#         self.params = params
#         self.mode = mode

#         # ========================================================================
#         # 기본 게인 값 (튜닝)
#         # ========================================================================
#         self.Kp_pos = np.array([10.0, 10.0, 15.0])
#         self.Kd_pos = np.array([5.0, 5.0, 8.0])
        
#         self.Kp_angle = np.array([100.0, 100.0, 100.0])
#         self.Kd_angle = np.array([10.0, 10.0, 10.0])

#     def reset(self) -> None:
#         """Reset controller state"""
#         pass

#     def __call__(self, t: float, s: np.ndarray, s_des: np.ndarray) -> tuple[float, np.ndarray]:
#         m = self.params.mass
#         g = self.params.grav
#         I = self.params.I

#         # ========================================================================
#         # 1. Parse Current & Desired States
#         # ========================================================================
#         pos = s[0:3]
#         vel = s[3:6]
#         quat = s[6:10]
#         omega = s[10:13] # Body angular velocities [p, q, r]

#         pos_des = s_des[0:3]
#         vel_des = s_des[3:6]
#         acc_des = s_des[6:9]
#         psi_des = s_des[9]
#         psi_dot_des = s_des[10]

#         R = quaternion_to_R(quat)
#         phi, theta, psi = rot_to_rpy_zxy(R)

#         # ========================================================================
#         # 2. Position Controller
#         # ========================================================================
#         e_pos = pos_des - pos
#         e_vel = vel_des - vel
        
#         # Feedforward / Kp-only 모드 구분
#         if self.mode == "no_forward":
#             acc_c = self.Kp_pos * e_pos + self.Kd_pos * e_vel
#         elif self.mode == "no_forward_kp_only":
#             acc_c = self.Kp_pos * e_pos
#         else:
#             acc_c = acc_des + self.Kp_pos * e_pos + self.Kd_pos * e_vel

#         # model based라서 기본적으로 줘야하는 중력 들어있음
#         F = m * (g + acc_c[2]) 

#         # ========================================================================
#         # 3. Attitude Command Calculation
#         # ========================================================================
#         # 가속도 명령을 위해 달성해야하는 롤피치
#         phi_c = (1.0 / g) * (acc_c[0] * np.sin(psi) - acc_c[1] * np.cos(psi))
#         theta_c = (1.0 / g) * (acc_c[0] * np.cos(psi) + acc_c[1] * np.sin(psi))

#         # ========================================================================
#         # 4. Attitude Controller
#         # ========================================================================
#         e_phi = wrap_to_pi(phi_c - phi)
#         e_theta = wrap_to_pi(theta_c - theta)
#         e_psi = wrap_to_pi(psi_des - psi)
        
#         ang_pos_err = np.array([e_phi, e_theta, e_psi])

#         # Feedforward의 유무 모드 구분
#         if self.mode == "no_forward":
#             ang_rate_err = np.array([0.0, 0.0, 0.0]) - omega
#         else:
#             ang_rate_err = np.array([0.0, 0.0, psi_dot_des]) - omega

#         alpha_c = self.Kp_angle * ang_pos_err + self.Kd_angle * ang_rate_err

#         # 관성 모멘트 보상 역시 모드에 상관없이 항상 해줍니다.
#         M = I @ alpha_c + np.cross(omega, I @ omega)

#         return F, M

























from __future__ import annotations

import numpy as np
from typing import Tuple

from .math_utils import quaternion_to_R, rot_to_rpy_zxy, wrap_to_pi
from .model import QuadParams


def _hat(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def _vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)


def _safe_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _normalize_with_derivatives(
    v: np.ndarray,
    v_dot: np.ndarray,
    v_ddot: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    u = v / ||v||
    returns: u, u_dot, u_ddot
    """
    n = np.linalg.norm(v)
    if n < eps:
        u = np.array([0.0, 0.0, 1.0], dtype=float)
        return u, np.zeros(3), np.zeros(3)

    u = v / n
    n_dot = u @ v_dot
    u_dot = (v_dot - u * n_dot) / n

    n_ddot = (u_dot @ v_dot) + (u @ v_ddot)
    u_ddot = (v_ddot - u * n_ddot - 2.0 * u_dot * n_dot) / n

    return u, u_dot, u_ddot


class Controller:
    """
    Mellinger-structure controller, stabilized for assignment/simulator use.

    Required desired state layout:
      s_des =
      [x, y, z,
       vx, vy, vz,
       ax, ay, az,
       jx, jy, jz,
       sx, sy, sz,
       psi, psi_dot, psi_ddot]

    Current state layout:
      s =
      [x, y, z,
       vx, vy, vz,
       qw, qx, qy, qz,
       p, q, r]
    """

    def __init__(self, params: QuadParams, mode: str = "pd") -> None:
        self.params = params
        self.mode = mode.lower()

        # PD mode gains
        self.Kp_pos = np.array([10.0, 10.0, 15.0], dtype=float)
        self.Kd_pos = np.array([5.0, 5.0, 8.0], dtype=float)

        # PD mode attitude gains
        self.Kp_angle = np.array([100.0, 100.0, 100.0], dtype=float)
        self.Kd_angle = np.array([10.0, 10.0, 10.0], dtype=float)

        # Geometric mode gains
        self.KR = np.array([2.0, 2.0, 1.0], dtype=float)
        self.KOmega = np.array([0.20, 0.20, 0.15], dtype=float)
        self.use_projection_thrust = True

        # Saturation limits
        self.max_force_scale = 2.5
        self.max_torque = np.array([0.6, 0.6, 0.3], dtype=float)

    def reset(self) -> None:
        pass

    def __call__(self, t: float, s: np.ndarray, s_des: np.ndarray) -> tuple[float, np.ndarray]:
        if len(s_des) < 18:
            raise ValueError(
                "This controller requires s_des = "
                "[pos(3), vel(3), acc(3), jerk(3), snap(3), psi, psi_dot, psi_ddot]"
            )

        if self.mode in {"pd", "model_based"}:
            return self._call_pd(t, s, s_des)
        if self.mode in {"geometric", "mellinger"}:
            return self._call_geometric(t, s, s_des)

        raise ValueError(f"Unsupported controller mode: {self.mode}")

    def _call_pd(self, t: float, s: np.ndarray, s_des: np.ndarray) -> tuple[float, np.ndarray]:
        del t

        m = float(self.params.mass)
        g = float(self.params.grav)
        I = np.asarray(self.params.I, dtype=float)

        # 1. Parse current state
        pos = np.asarray(s[0:3], dtype=float)
        vel = np.asarray(s[3:6], dtype=float)
        quat = np.asarray(s[6:10], dtype=float)
        omega = np.asarray(s[10:13], dtype=float)

        # 2. Parse desired state (18D format; jerk/snap are currently unused)
        pos_des = np.asarray(s_des[0:3], dtype=float)
        vel_des = np.asarray(s_des[3:6], dtype=float)
        acc_des = np.asarray(s_des[6:9], dtype=float)
        psi_des = float(s_des[15])
        psi_dot_des = float(s_des[16])

        R = quaternion_to_R(quat)
        phi, theta, psi = rot_to_rpy_zxy(R)

        # 3. Position control with feed-forward acceleration
        e_pos = pos_des - pos
        e_vel = vel_des - vel
        acc_cmd = acc_des + self.Kp_pos * e_pos + self.Kd_pos * e_vel

        # Model-based thrust
        F = m * (g + acc_cmd[2])

        # 4. Desired roll/pitch from lateral acceleration command
        phi_c = (acc_cmd[0] * np.sin(psi) - acc_cmd[1] * np.cos(psi)) / g
        theta_c = (acc_cmd[0] * np.cos(psi) + acc_cmd[1] * np.sin(psi)) / g

        # 5. Attitude control
        e_phi = wrap_to_pi(phi_c - phi)
        e_theta = wrap_to_pi(theta_c - theta)
        e_psi = wrap_to_pi(psi_des - psi)
        ang_pos_err = np.array([e_phi, e_theta, e_psi], dtype=float)

        ang_rate_err = np.array([0.0, 0.0, psi_dot_des], dtype=float) - omega
        alpha_cmd = self.Kp_angle * ang_pos_err + self.Kd_angle * ang_rate_err

        M = I @ alpha_cmd + np.cross(omega, I @ omega)

        # 6. Saturation and safety guards
        F = float(np.clip(F, 0.0, self.max_force_scale * m * g))
        M = np.clip(M, -self.max_torque, self.max_torque)

        if not np.isfinite(F):
            F = 0.0
        if not np.all(np.isfinite(M)):
            M = np.zeros(3, dtype=float)

        return F, M

    def _call_geometric(self, t: float, s: np.ndarray, s_des: np.ndarray) -> tuple[float, np.ndarray]:
        del t

        m = float(self.params.mass)
        g = float(self.params.grav)
        J = np.asarray(self.params.I, dtype=float)

        pos = np.asarray(s[0:3], dtype=float)
        vel = np.asarray(s[3:6], dtype=float)
        quat = np.asarray(s[6:10], dtype=float)
        omega = np.asarray(s[10:13], dtype=float)

        # Simulator dynamics convention uses the transpose for body->world mapping here.
        R = quaternion_to_R(quat).T
        e3 = np.array([0.0, 0.0, 1.0], dtype=float)

        pos_des = np.asarray(s_des[0:3], dtype=float)
        vel_des = np.asarray(s_des[3:6], dtype=float)
        acc_des = np.asarray(s_des[6:9], dtype=float)
        jerk_des = np.asarray(s_des[9:12], dtype=float)
        snap_des = np.asarray(s_des[12:15], dtype=float)
        psi_des = float(s_des[15])
        psi_dot_des = float(s_des[16])
        psi_ddot_des = float(s_des[17])

        e_pos = pos_des - pos
        e_vel = vel_des - vel

        a_fb = self.Kp_pos * e_pos + self.Kd_pos * e_vel
        a_cmd = acc_des + a_fb

        A = a_cmd + g * e3
        A_dot = jerk_des
        A_ddot = snap_des

        if np.linalg.norm(A) < 1e-8:
            A = g * e3
            A_dot = np.zeros(3, dtype=float)
            A_ddot = np.zeros(3, dtype=float)

        b3d, b3d_dot, b3d_ddot = _normalize_with_derivatives(A, A_dot, A_ddot)

        b1c = np.array([np.cos(psi_des), np.sin(psi_des), 0.0], dtype=float)
        b1c_dot = np.array(
            [-np.sin(psi_des) * psi_dot_des, np.cos(psi_des) * psi_dot_des, 0.0],
            dtype=float,
        )
        b1c_ddot = np.array(
            [
                -np.cos(psi_des) * psi_dot_des**2 - np.sin(psi_des) * psi_ddot_des,
                -np.sin(psi_des) * psi_dot_des**2 + np.cos(psi_des) * psi_ddot_des,
                0.0,
            ],
            dtype=float,
        )

        c2 = np.cross(b3d, b1c)
        c2_dot = np.cross(b3d_dot, b1c) + np.cross(b3d, b1c_dot)
        c2_ddot = (
            np.cross(b3d_ddot, b1c)
            + 2.0 * np.cross(b3d_dot, b1c_dot)
            + np.cross(b3d, b1c_ddot)
        )

        if np.linalg.norm(c2) < 1e-8:
            c2 = np.array([0.0, 1.0, 0.0], dtype=float)
            c2_dot = np.zeros(3, dtype=float)
            c2_ddot = np.zeros(3, dtype=float)

        b2d, b2d_dot, b2d_ddot = _normalize_with_derivatives(c2, c2_dot, c2_ddot)

        b1d = np.cross(b2d, b3d)
        b1d_dot = np.cross(b2d_dot, b3d) + np.cross(b2d, b3d_dot)
        b1d_ddot = (
            np.cross(b2d_ddot, b3d)
            + 2.0 * np.cross(b2d_dot, b3d_dot)
            + np.cross(b2d, b3d_ddot)
        )

        b1d = _safe_normalize(b1d)
        b2d = _safe_normalize(np.cross(b3d, b1d))
        b3d = _safe_normalize(b3d)

        R_des = np.column_stack((b1d, b2d, b3d))
        R_des_dot = np.column_stack((b1d_dot, b2d_dot, b3d_dot))
        R_des_ddot = np.column_stack((b1d_ddot, b2d_ddot, b3d_ddot))

        omega_des_hat = R_des.T @ R_des_dot
        omega_des = _vee(omega_des_hat)

        omega_des_hat_dot = R_des.T @ R_des_ddot - omega_des_hat @ omega_des_hat
        omega_des_dot = _vee(omega_des_hat_dot)

        omega_des = np.clip(omega_des, -5.0, 5.0)
        omega_des_dot = np.clip(omega_des_dot, -20.0, 20.0)

        F_des = m * A
        b3 = R[:, 2]

        if self.use_projection_thrust:
            F = float(F_des @ b3)
        else:
            F = float(np.linalg.norm(F_des))

        e_R_mat = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = _vee(e_R_mat)

        e_omega = omega - (R.T @ R_des @ omega_des)

        M = (
            -self.KR * e_R
            -self.KOmega * e_omega
            + np.cross(omega, J @ omega)
            - J
            @ (
                _hat(omega) @ (R.T @ R_des @ omega_des)
                - (R.T @ R_des @ omega_des_dot)
            )
        )

        F = float(np.clip(F, 0.0, self.max_force_scale * m * g))
        M = np.clip(M, -self.max_torque, self.max_torque)

        if not np.isfinite(F):
            F = 0.0
        if not np.all(np.isfinite(M)):
            M = np.zeros(3, dtype=float)

        return F, M