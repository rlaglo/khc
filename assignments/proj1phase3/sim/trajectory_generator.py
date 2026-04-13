from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np
from scipy.linalg import block_diag

def _generate_minimum_unconstrained(
    waypoints: np.ndarray,
    n_seg: int,
    total_time: float,
    objective: str = "snap",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Richter et al. style closed-form unconstrained QP.
    Current implementation keeps 7th-order polynomial and [p,v,a,j] endpoint derivatives.

    objective:
        - "snap": minimize integral of squared 4th derivative
        - "jerk": minimize integral of squared 3rd derivative
    """
    T_scale = _get_time_allocation(waypoints, total_time)
    n_coeffs = 8  # keep 7th-order polynomial

    if objective == "snap":
        k_cost = 4
    elif objective == "jerk":
        k_cost = 3
    else:
        raise ValueError("objective must be 'jerk' or 'snap'")

    # Keep derivative-state structure [p, v, a, j] at each end.
    k_r = 4

    Q_list = []
    M_list = []

    for m in range(n_seg):
        Tm = T_scale[m]

        # Cost block Q_m for chosen derivative order
        Q_m = np.zeros((n_coeffs, n_coeffs))
        for i in range(k_cost, n_coeffs):
            for j in range(k_cost, n_coeffs):
                ci = math.factorial(i) / math.factorial(i - k_cost)
                cj = math.factorial(j) / math.factorial(j - k_cost)
                power = i + j - 2 * k_cost + 1
                Q_m[i, j] = 2.0 * ci * cj * (Tm ** power) / power
        Q_list.append(Q_m)

        # Mapping from polynomial coeffs to endpoint derivatives [p,v,a,j] at t=0 and t=Tm
        M_m = np.zeros((n_coeffs, n_coeffs))

        # start derivatives at t=0
        M_m[0, 0] = 1.0  # p(0)
        if n_coeffs >= 2:
            M_m[1, 1] = 1.0  # v(0)
        if n_coeffs >= 3:
            M_m[2, 2] = 2.0  # a(0)
        if n_coeffs >= 4:
            M_m[3, 3] = 6.0  # j(0)

        # end derivatives at t=Tm
        for i in range(n_coeffs):
            M_m[4, i] = Tm ** i
            if i >= 1:
                M_m[5, i] = i * (Tm ** (i - 1))
            if i >= 2:
                M_m[6, i] = i * (i - 1) * (Tm ** (i - 2))
            if i >= 3:
                M_m[7, i] = i * (i - 1) * (i - 2) * (Tm ** (i - 3))

        M_list.append(M_m)

    Q = block_diag(*Q_list)
    M = block_diag(*M_list)
    M_inv = np.linalg.inv(M)

    # Selection matrix: d_all = C [d_C; d_U]
    C, num_C = _build_selection_matrix(n_seg, k_r=k_r)

    # Reduced cost matrix
    R = C.T @ M_inv.T @ Q @ M_inv @ C

    R_UU = R[num_C:, num_C:]
    R_UC = R[num_C:, :num_C]

    c_x = _solve_1d_unconstrained(waypoints[:, 0], R_UU, R_UC, M_inv, C, num_C, n_seg, n_coeffs)
    c_y = _solve_1d_unconstrained(waypoints[:, 1], R_UU, R_UC, M_inv, C, num_C, n_seg, n_coeffs)
    c_z = _solve_1d_unconstrained(waypoints[:, 2], R_UU, R_UC, M_inv, C, num_C, n_seg, n_coeffs)

    return c_x, c_y, c_z, T_scale

# (참고: _build_selection_matrix 와 _solve_1d_unconstrained 는 인덱싱 배열을 
# d_C 와 d_U 로 쪼개고 결합하는 유틸리티 함수로 별도 구현이 필요합니다.)

def _build_selection_matrix(n_seg: int, k_r: int = 4) -> tuple[np.ndarray, int]:
    """
    모든 세그먼트의 미분값(d_all)을 고정 변수(d_C)와 자유 변수(d_U)로 
    매핑하는 선택 행렬 C와 고정 변수의 개수를 반환합니다.
    """
    # 각 세그먼트당 시작/종료점 x (위치, 속도, 가속도, 저크)
    num_all = 2 * k_r * n_seg 
    
    # 고정 변수(d_C): 모든 경유지 위치 + 시작점(v,a,j) + 종료점(v,a,j)
    num_C = (n_seg + 1) + 2 * (k_r - 1)  
    
    # 자유 변수(d_U): 중간 경유지들의 (v,a,j)
    num_U = (n_seg - 1) * (k_r - 1)      
    num_total = num_C + num_U

    # 매핑 행렬 (d_all = C * [d_C, d_U]^T)
    C = np.zeros((num_all, num_total))

    for m in range(n_seg):
        # 1. 세그먼트 m의 시작점 (t = 0)
        C[m * 2 * k_r + 0, m] = 1.0  # 위치 p_m (고정)
        
        for d in range(1, k_r):
            if m == 0:
                # 전체 궤적의 시작 경계 조건 (고정)
                C[m * 2 * k_r + d, (n_seg + 1) + (d - 1)] = 1.0
            else:
                # 중간 경유지의 연속된 미분값 (자유)
                C[m * 2 * k_r + d, num_C + (m - 1) * (k_r - 1) + (d - 1)] = 1.0

        # 2. 세그먼트 m의 종료점 (t = Tm)
        C[m * 2 * k_r + k_r + 0, m + 1] = 1.0  # 위치 p_{m+1} (고정)
        
        for d in range(1, k_r):
            if m == n_seg - 1:
                # 전체 궤적의 종료 경계 조건 (고정)
                C[m * 2 * k_r + k_r + d, (n_seg + 1) + (k_r - 1) + (d - 1)] = 1.0
            else:
                # 중간 경유지의 연속된 미분값 (자유)
                C[m * 2 * k_r + k_r + d, num_C + m * (k_r - 1) + (d - 1)] = 1.0

    return C, num_C

def _solve_1d_unconstrained(
    wp_1d: np.ndarray,
    R_UU: np.ndarray,
    R_UC: np.ndarray,
    M_inv: np.ndarray,
    C: np.ndarray,
    num_C: int,
    n_seg: int,
    n_coeffs: int,
) -> np.ndarray:
    d_C = np.zeros(num_C)
    d_C[0:n_seg+1] = wp_1d

    if R_UU.shape[0] > 0:
        d_U = np.linalg.solve(R_UU, -R_UC @ d_C)
    else:
        d_U = np.zeros(0)

    d = np.concatenate((d_C, d_U))
    d_all = C @ d
    c_flat = M_inv @ d_all
    return c_flat.reshape((n_seg, n_coeffs)).T

def _poly_derivative_asc(coeffs: np.ndarray) -> np.ndarray:
    """
    coeffs are in ascending order:
        p(t) = c0 + c1 t + c2 t^2 + ...
    returns derivative coeffs in ascending order.
    """
    degree = len(coeffs) - 1
    if degree <= 0:
        return np.array([0.0], dtype=float)
    return np.array([(i + 1) * coeffs[i + 1] for i in range(degree)], dtype=float)


def _poly_eval_asc(coeffs: np.ndarray, t: float) -> float:
    """Evaluate ascending-order polynomial at t."""
    out = 0.0
    tp = 1.0
    for c in coeffs:
        out += c * tp
        tp *= t
    return float(out)


def _solve_equality_qp(H: np.ndarray, Aeq: np.ndarray, beq: np.ndarray) -> np.ndarray:
    """
    Solve:
        min 0.5 x^T H x
        s.t. Aeq x = beq
    using KKT system.
    """
    H = 0.5 * (H + H.T)
    n = H.shape[0]
    m = Aeq.shape[0]

    KKT = np.zeros((n + m, n + m), dtype=float)
    KKT[:n, :n] = H
    KKT[:n, n:] = Aeq.T
    KKT[n:, :n] = Aeq

    rhs = np.zeros(n + m, dtype=float)
    rhs[n:] = beq

    sol, _, _, _ = np.linalg.lstsq(KKT, rhs, rcond=None)
    return sol[:n]


def _get_time_allocation(
    waypoints: np.ndarray,
    total_time: float,
    min_segment_time: float = 0.8,
) -> np.ndarray:
    dists = np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1)
    n_seg = len(dists)

    if n_seg == 0:
        return np.array([], dtype=float)

    weights = np.maximum(dists, 1e-6)
    weights = weights / np.sum(weights)

    t_min_total = min_segment_time * n_seg
    if total_time <= t_min_total:
        return np.full(n_seg, min_segment_time, dtype=float)

    extra = total_time - t_min_total
    return min_segment_time + extra * weights


def _remove_duplicate_waypoints(waypoints: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if len(waypoints) <= 1:
        return waypoints.copy()

    filtered = [waypoints[0]]
    for i in range(1, len(waypoints)):
        if np.linalg.norm(waypoints[i] - filtered[-1]) > eps:
            filtered.append(waypoints[i])

    return np.asarray(filtered, dtype=float)


def _generate_smooth_only(
    waypoints: np.ndarray,
    n_seg: int,
    total_time: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth-only trajectory with 5th-order (quintic) polynomial per segment.
    C^2 continuous (position, velocity, acceleration) but no cost minimization.
    Uses least-squares to find coefficient solution.
    """
    T_scale = _get_time_allocation(waypoints, total_time, min_segment_time=0.8)

    n_coeffs = 6  # 5th-order polynomial
    V = n_seg * n_coeffs

    Aeq_list = []
    beq_x, beq_y, beq_z = [], [], []

    # 1. Segment start/end position constraints
    for m in range(n_seg):
        Tm = T_scale[m]

        # start position at t=0
        row_start = np.zeros(V, dtype=float)
        row_start[m * n_coeffs + 0] = 1.0
        Aeq_list.append(row_start)
        beq_x.append(waypoints[m, 0])
        beq_y.append(waypoints[m, 1])
        beq_z.append(waypoints[m, 2])

        # end position at t=Tm
        row_end = np.zeros(V, dtype=float)
        for i in range(n_coeffs):
            row_end[m * n_coeffs + i] = Tm ** i
        Aeq_list.append(row_end)
        beq_x.append(waypoints[m + 1, 0])
        beq_y.append(waypoints[m + 1, 1])
        beq_z.append(waypoints[m + 1, 2])

    # 2. C^2 continuity at internal waypoints (velocity, acceleration)
    for m in range(n_seg - 1):
        Tm = T_scale[m]
        for deriv in range(1, 3):  # velocity and acceleration
            row = np.zeros(V, dtype=float)
            # left side derivative at Tm
            for i in range(deriv, n_coeffs):
                row[m * n_coeffs + i] = (
                    math.factorial(i) / math.factorial(i - deriv) * (Tm ** (i - deriv))
                )
            # right side derivative at 0
            row[(m + 1) * n_coeffs + deriv] = -math.factorial(deriv)
            Aeq_list.append(row)
            beq_x.append(0.0)
            beq_y.append(0.0)
            beq_z.append(0.0)

    # 3. Zero boundary derivatives at start/end (velocity and acceleration)
    for deriv in range(1, 3):
        # start
        row_start = np.zeros(V, dtype=float)
        row_start[deriv] = math.factorial(deriv)
        Aeq_list.append(row_start)
        beq_x.append(0.0)
        beq_y.append(0.0)
        beq_z.append(0.0)

        # end
        row_end = np.zeros(V, dtype=float)
        Tm = T_scale[-1]
        for i in range(deriv, n_coeffs):
            row_end[(n_seg - 1) * n_coeffs + i] = (
                math.factorial(i) / math.factorial(i - deriv) * (Tm ** (i - deriv))
            )
        Aeq_list.append(row_end)
        beq_x.append(0.0)
        beq_y.append(0.0)
        beq_z.append(0.0)

    Aeq = np.array(Aeq_list, dtype=float)

    # For smooth_only, we don't minimize any cost, just satisfy constraints
    # Use least-squares to find minimum-norm solution
    c_x_flat, _, _, _ = np.linalg.lstsq(Aeq, np.array(beq_x, dtype=float), rcond=None)
    c_y_flat, _, _, _ = np.linalg.lstsq(Aeq, np.array(beq_y, dtype=float), rcond=None)
    c_z_flat, _, _, _ = np.linalg.lstsq(Aeq, np.array(beq_z, dtype=float), rcond=None)

    c_x = c_x_flat.reshape(n_seg, n_coeffs).T
    c_y = c_y_flat.reshape(n_seg, n_coeffs).T
    c_z = c_z_flat.reshape(n_seg, n_coeffs).T

    return c_x, c_y, c_z, T_scale


def _generate_minimum_snap(
    waypoints: np.ndarray,
    n_seg: int,
    total_time: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimum-snap trajectory with 7th-order polynomial per segment.
    """
    T_scale = _get_time_allocation(waypoints, total_time, min_segment_time=0.8)

    n_coeffs = 8   # 7th-order polynomial
    d = 4          # snap (4th derivative)
    V = n_seg * n_coeffs

    # Cost matrix for integral of squared snap
    H = np.zeros((V, V), dtype=float)
    for m in range(n_seg):
        Tm = T_scale[m]
        for i in range(d, n_coeffs):
            for j in range(d, n_coeffs):
                ci = math.factorial(i) / math.factorial(i - d)
                cj = math.factorial(j) / math.factorial(j - d)
                power = i + j - 2 * d + 1
                H[m * n_coeffs + i, m * n_coeffs + j] = 2.0 * ci * cj * (Tm ** power) / power

    Aeq_list = []
    beq_x, beq_y, beq_z = [], [], []

    # 1. Segment start/end position constraints
    for m in range(n_seg):
        Tm = T_scale[m]

        row_start = np.zeros(V, dtype=float)
        row_start[m * n_coeffs + 0] = 1.0
        Aeq_list.append(row_start)
        beq_x.append(waypoints[m, 0])
        beq_y.append(waypoints[m, 1])
        beq_z.append(waypoints[m, 2])

        row_end = np.zeros(V, dtype=float)
        for i in range(n_coeffs):
            row_end[m * n_coeffs + i] = Tm ** i
        Aeq_list.append(row_end)
        beq_x.append(waypoints[m + 1, 0])
        beq_y.append(waypoints[m + 1, 1])
        beq_z.append(waypoints[m + 1, 2])

    # 2. Continuity at internal waypoints for derivatives 1 to 3 (Velocity, Accel, Jerk)
    # 물리적 연속성 보장을 위해 Jerk(3차) 까지만 강제하고, Snap(4차) 이상은 H행렬이 최적화하도록 자유도를 부여합니다.
    for m in range(n_seg - 1):
        Tm = T_scale[m]
        for c in range(1, 4):  # 수정됨: range(1, 7) -> range(1, 4)
            row = np.zeros(V, dtype=float)
            for i in range(c, n_coeffs):
                row[m * n_coeffs + i] = (math.factorial(i) / math.factorial(i - c)) * (Tm ** (i - c))

            row[(m + 1) * n_coeffs + c] = -math.factorial(c)
            Aeq_list.append(row)
            beq_x.append(0.0)
            beq_y.append(0.0)
            beq_z.append(0.0)

    # 3. Zero boundary derivatives at start/end for derivatives 1 to 4 (Vel, Acc, Jerk, Snap)
    # Wiggling 방지를 위해 경계면에서는 Snap(4차)까지 완벽히 0으로 묶어버립니다.
    for c in range(1, 5):  # 수정됨: range(1, 4) -> range(1, 5)
        row_start = np.zeros(V, dtype=float)
        row_start[c] = math.factorial(c)
        Aeq_list.append(row_start)
        beq_x.append(0.0)
        beq_y.append(0.0)
        beq_z.append(0.0)

        row_end = np.zeros(V, dtype=float)
        Tm = T_scale[-1]
        for i in range(c, n_coeffs):
            row_end[(n_seg - 1) * n_coeffs + i] = (math.factorial(i) / math.factorial(i - c)) * (Tm ** (i - c))
        Aeq_list.append(row_end)
        beq_x.append(0.0)
        beq_y.append(0.0)
        beq_z.append(0.0)

    Aeq = np.array(Aeq_list, dtype=float)

    # KKT System Solver
    c_x_flat = _solve_equality_qp(H, Aeq, np.array(beq_x, dtype=float))
    c_y_flat = _solve_equality_qp(H, Aeq, np.array(beq_y, dtype=float))
    c_z_flat = _solve_equality_qp(H, Aeq, np.array(beq_z, dtype=float))

    c_x = c_x_flat.reshape(n_seg, n_coeffs).T
    c_y = c_y_flat.reshape(n_seg, n_coeffs).T
    c_z = c_z_flat.reshape(n_seg, n_coeffs).T

    return c_x, c_y, c_z, T_scale


@dataclass
class TrajectoryGenerator:
    waypoints: np.ndarray
    method: str = "snap"
    total_time: float = 25.0

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError("waypoints must be shaped (N, 3)")

        self.waypoints = _remove_duplicate_waypoints(self.waypoints)

        self.n_seg = self.waypoints.shape[0] - 1
        if self.n_seg < 1:
            raise ValueError("need at least two distinct waypoints")

        if self.method not in {"smooth", "jerk", "snap"}:
            raise ValueError("method must be 'smooth', 'jerk', or 'snap'")

        if self.method == "smooth":
            # true smooth-only: 5th-order quintic with C^2 continuity, no optimization
            self.c_x, self.c_y, self.c_z, self.T_scale = _generate_smooth_only(
                self.waypoints,
                self.n_seg,
                self.total_time,
            )
        elif self.method == "jerk":
            # minimum-jerk: 7th-order with jerk minimization
            self.c_x, self.c_y, self.c_z, self.T_scale = _generate_minimum_unconstrained(
                self.waypoints,
                self.n_seg,
                self.total_time,
                objective="jerk",
            )
        elif self.method == "snap":
            # minimum-snap: 7th-order with snap minimization
            self.c_x, self.c_y, self.c_z, self.T_scale = _generate_minimum_snap(
                self.waypoints,
                self.n_seg,
                self.total_time,
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        self.total_duration = float(np.sum(self.T_scale))

    def _segment_time(self, t: float) -> Tuple[int, float]:
        t = float(t)

        if t <= 0.0:
            return 0, 0.0

        if t >= self.total_duration:
            return self.n_seg - 1, self.T_scale[-1]

        rem = t
        for seg in range(self.n_seg):
            if rem <= self.T_scale[seg]:
                return seg, rem
            rem -= self.T_scale[seg]

        return self.n_seg - 1, self.T_scale[-1]

    def evaluate(self, t: float) -> np.ndarray:
        """
        Returns controller-compatible desired state:
        s_des = [x, y, z,
                 vx, vy, vz,
                 ax, ay, az,
                 jx, jy, jz,
                 sx, sy, sz,
                 psi, psi_dot, psi_ddot]
        """
        seg, local_t = self._segment_time(t)

        coeff_x = self.c_x[:, seg]
        coeff_y = self.c_y[:, seg]
        coeff_z = self.c_z[:, seg]

        d1_x = _poly_derivative_asc(coeff_x)
        d1_y = _poly_derivative_asc(coeff_y)
        d1_z = _poly_derivative_asc(coeff_z)

        d2_x = _poly_derivative_asc(d1_x)
        d2_y = _poly_derivative_asc(d1_y)
        d2_z = _poly_derivative_asc(d1_z)

        d3_x = _poly_derivative_asc(d2_x)
        d3_y = _poly_derivative_asc(d2_y)
        d3_z = _poly_derivative_asc(d2_z)

        d4_x = _poly_derivative_asc(d3_x)
        d4_y = _poly_derivative_asc(d3_y)
        d4_z = _poly_derivative_asc(d3_z)

        px = _poly_eval_asc(coeff_x, local_t)
        py = _poly_eval_asc(coeff_y, local_t)
        pz = _poly_eval_asc(coeff_z, local_t)

        vx = _poly_eval_asc(d1_x, local_t)
        vy = _poly_eval_asc(d1_y, local_t)
        vz = _poly_eval_asc(d1_z, local_t)

        ax = _poly_eval_asc(d2_x, local_t)
        ay = _poly_eval_asc(d2_y, local_t)
        az = _poly_eval_asc(d2_z, local_t)

        # Keep yaw fixed to zero for robustness.
        psi = 0.0
        psi_dot = 0.0
        psi_ddot = 0.0

        jx = _poly_eval_asc(d3_x, local_t)
        jy = _poly_eval_asc(d3_y, local_t)
        jz = _poly_eval_asc(d3_z, local_t)

        sx = _poly_eval_asc(d4_x, local_t)
        sy = _poly_eval_asc(d4_y, local_t)
        sz = _poly_eval_asc(d4_z, local_t)

        s_des = np.zeros(18, dtype=float)

        s_des[0:3] = [px, py, pz]
        s_des[3:6] = [vx, vy, vz]
        s_des[6:9] = [ax, ay, az]
        s_des[9:12] = [jx, jy, jz]
        s_des[12:15] = [sx, sy, sz]
        s_des[15] = psi
        s_des[16] = psi_dot
        s_des[17] = psi_ddot

        return s_des






