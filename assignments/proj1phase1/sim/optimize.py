"""Gain optimization module for quadrotor controller.

[ADDED] This module provides automated gain optimization using Ziegler-Nichols tuning method.
It empirically identifies critical system parameters and applies Z-N tuning rules for PD gains.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

from .controller import Controller
from .dynamics import quad_eom
from .math_utils import quaternion_to_R, rot_to_rpy_zxy, wrap_to_pi
from .model import QuadParams
from .simulator import SimulationEngine, SimulationResult, initial_state, rk4_step


@dataclass
class OptimizationResult:
    """Results from gain optimization."""
    best_gains: dict[str, np.ndarray]
    best_metrics: dict[str, float]
    optimization_history: list[dict]
    num_evaluations: int


# [ADDED] Fast non-historical simulator for optimization

class FastSimulationEngine(SimulationEngine):
    """SimulationEngine that skips history recording and computes metrics on the fly."""

    def reset(self) -> None:
        self.t = 0.0
        self.state = initial_state()
        self.F = self.params.mass * self.params.grav
        self.M = np.zeros(3)
        self.done = False
        self.controller.reset()
        self._rng = np.random.default_rng(self.seed)

        self._pos_error_sq = 0.0
        self._vel_error_sq = 0.0
        self._yaw_error_sq = 0.0
        self._count = 0
        self._smooth_dF_sq = 0.0
        self._smooth_dM_sq = 0.0
        self._smooth_count = 0
        self._prev_F = self.F
        self._prev_M = self.M
        self._has_prev = False
        self._track_zero_crossings = False
        self._last_z_sign = None
        self._zero_crossing_times: list[float] = []

    def step(self, controls: int = 1) -> None:
        if self.done:
            return

        for _ in range(max(1, int(controls))):
            if self.t >= self.t_final - 1e-12:
                self.done = True
                break

            Fd = self._rng.normal(0.0, self.fnoise, 3)
            for _ in range(self.steps_per_control):
                self.state = rk4_step(
                    quad_eom,
                    self.t,
                    self.state,
                    self.t_step,
                    self.F,
                    self.M,
                    Fd,
                    self.params,
                )
                self.t += self.t_step

            desired = self.trajectory_fn(self.t, self.state)
            self.F, self.M = self.controller(self.t, self.state, desired)
            self._accumulate_metrics(desired)

            if self.t >= self.t_final - 1e-12:
                self.done = True
                break

    def _accumulate_metrics(self, desired: np.ndarray) -> None:
        pos_err = self.state[0:3] - desired[0:3]
        vel_err = self.state[3:6] - desired[3:6]
        yaw = rot_to_rpy_zxy(quaternion_to_R(self.state[6:10]))[2]
        yaw_err = wrap_to_pi(yaw - desired[9])

        self._pos_error_sq += np.dot(pos_err, pos_err)
        self._vel_error_sq += np.dot(vel_err, vel_err)
        self._yaw_error_sq += yaw_err * yaw_err
        self._count += 1

        if self._track_zero_crossings:
            current_z_sign = np.sign(pos_err[2])
            if current_z_sign == 0:
                current_z_sign = 1.0
            if self._last_z_sign is not None and current_z_sign != self._last_z_sign:
                self._zero_crossing_times.append(self.t)
            self._last_z_sign = current_z_sign

        if self._has_prev:
            dF = self.F - self._prev_F
            dM = self.M - self._prev_M
            self._smooth_dF_sq += np.dot(dF, dF)
            self._smooth_dM_sq += np.dot(dM, dM)
            self._smooth_count += 1

        self._prev_F = self.F
        self._prev_M = self.M
        self._has_prev = True

    def get_metrics(self) -> tuple[float, float, float, float]:
        if self._count == 0:
            return 0.0, 0.0, 0.0, 0.0

        rmse_pos = np.sqrt(self._pos_error_sq / self._count)
        rmse_vel = np.sqrt(self._vel_error_sq / self._count)
        rmse_yaw = np.degrees(np.sqrt(self._yaw_error_sq / self._count))
        if self._smooth_count > 0:
            smoothness = (
                np.sqrt(self._smooth_dF_sq / self._smooth_count)
                + 10.0 * np.sqrt(self._smooth_dM_sq / self._smooth_count)
            )
        else:
            smoothness = 0.0

        return float(rmse_pos), float(rmse_vel), float(rmse_yaw), float(smoothness)


# [ADDED] Main optimizer class for gain tuning

class GainOptimizer:
    """Optimizer for quadrotor controller gains."""

    def __init__(
        self,
        trajectory_fn: Callable,
        params: QuadParams | None = None,
        t_final: float = 25.0,
        t_step: float = 0.002,
        control_step: float = 0.01,
        fnoise: float = 1.0,
        seed: int | None = None,
        controller_mode: str = "model_based",
    ):
        """Initialize optimizer.
        
        Args:
            trajectory_fn: Trajectory function for simulation
            params: Quadrotor parameters
            t_final: Simulation time
            t_step: Simulation step time
            control_step: Control step time
            fnoise: Noise standard deviation
            seed: Random seed
        """
        self.trajectory_fn = trajectory_fn
        self.params = params or QuadParams()
        self.t_final = t_final
        self.t_step = t_step
        self.control_step = control_step
        self.fnoise = fnoise
        self.seed = seed
        self.controller_mode = controller_mode
        
        self.evaluation_count = 0
        self.history = []
        self.progress_callback = None

    def _create_controller(self) -> Controller:
        return Controller(self.params, mode=self.controller_mode)

    def _run_simulation_with_gains(
        self,
        kp_pos: np.ndarray,
        kd_pos: np.ndarray,
        kp_angle: np.ndarray,
        kd_angle: np.ndarray,
    ) -> dict[str, float]:
        """Run simulation with given gains and return metrics.
        
        Returns:
            Dictionary with metrics: rmse_pos, rmse_vel, rmse_yaw_deg, smoothness
        """
        engine = FastSimulationEngine(
            trajectory_fn=self.trajectory_fn,
            params=self.params,
            t_final=self.t_final,
            t_step=self.t_step,
            control_step=self.control_step,
            fnoise=self.fnoise,
            seed=self.seed,
            controller=self._create_controller(),
        )
        engine.controller.Kp_pos = kp_pos.copy()
        engine.controller.Kd_pos = kd_pos.copy()
        engine.controller.Kp_angle = kp_angle.copy()
        engine.controller.Kd_angle = kd_angle.copy()

        while not engine.done:
            engine.step(controls=100)

        rmse_pos, rmse_vel, rmse_yaw, smoothness = engine.get_metrics()
        return {
            "rmse_pos": rmse_pos,
            "rmse_vel": rmse_vel,
            "rmse_yaw_deg": rmse_yaw,
            "smoothness": smoothness,
        }

    def optimize(
        self,
        kp_pos_ref: np.ndarray | None = None,
        kd_pos_ref: np.ndarray | None = None,
        kp_angle_ref: np.ndarray | None = None,
        kd_angle_ref: np.ndarray | None = None,
        margin: float = 0.5,
        max_iterations: int = 20,
        progress_callback: Callable | None = None,
    ) -> OptimizationResult:
        """Optimize controller gains using Ziegler-Nichols tuning method.
        
        [MODIFIED] Simplified to use only Ziegler-Nichols method.
        
        Args:
            kp_pos_ref: Reference Kp_pos (uses defaults if None)
            kd_pos_ref: Reference Kd_pos (uses defaults if None)
            kp_angle_ref: Reference Kp_angle (uses defaults if None)
            kd_angle_ref: Reference Kd_angle (uses defaults if None)
            margin: Not used (kept for API compatibility)
            max_iterations: Maximum iterations for critical parameter estimation
            progress_callback: Callback function for progress updates
            
        Returns:
            OptimizationResult with Z-N tuned gains
        """
        return self.optimize_ziegler_nichols(
            kp_pos_ref=kp_pos_ref,
            kd_pos_ref=kd_pos_ref,
            kp_angle_ref=kp_angle_ref,
            kd_angle_ref=kd_angle_ref,
            margin=margin,
            max_iterations=max_iterations,
            progress_callback=progress_callback,
        )

    def _zero_crossing_times(
        self,
        times: np.ndarray,
        signal: np.ndarray,
        transient_fraction: float = 0.2,
    ) -> np.ndarray:
        start = int(len(times) * transient_fraction)
        if start >= len(times) - 1:
            return np.zeros(0)

        trimmed_times = times[start:]
        trimmed_signal = signal[start:]
        signs = np.sign(trimmed_signal)
        signs[signs == 0] = 1
        crossing_indices = np.where(np.diff(signs) != 0)[0] + 1
        return trimmed_times[crossing_indices]

    def _estimate_ultimate_gain(
        self,
        kp_pos_ref: np.ndarray,
        kd_pos_ref: np.ndarray,
        kp_angle_ref: np.ndarray,
        kd_angle_ref: np.ndarray,
        max_steps: int = 8,
    ) -> tuple[float, float]:
        scales = [0.5, 1.0, 2.0, 4.0, 6.0]
        for scale in scales[:min(max_steps, len(scales))]:
            controller = self._create_controller()
            controller.Kp_pos = kp_pos_ref * scale
            controller.Kd_pos = np.zeros_like(kd_pos_ref)
            controller.Kp_angle = kp_angle_ref.copy()
            controller.Kd_angle = kd_angle_ref.copy()

            engine = FastSimulationEngine(
                trajectory_fn=self.trajectory_fn,
                params=self.params,
                t_final=self.t_final,
                t_step=self.t_step,
                control_step=self.control_step,
                fnoise=0.0,
                seed=self.seed,
                controller=controller,
            )
            engine._track_zero_crossings = True
            while not engine.done:
                engine.step(controls=100)

            crossings = engine._zero_crossing_times
            if len(crossings) >= 6:
                tu = 2.0 * float(np.mean(np.diff(crossings)))
                return float(scale), max(tu, 0.1)

        return float(scales[min(max_steps, len(scales)) - 1]), 2.0

    def optimize_ziegler_nichols(
        self,
        kp_pos_ref: np.ndarray | None = None,
        kd_pos_ref: np.ndarray | None = None,
        kp_angle_ref: np.ndarray | None = None,
        kd_angle_ref: np.ndarray | None = None,
        margin: float = 0.5,
        max_iterations: int = 15,
        progress_callback: Callable | None = None,
    ) -> OptimizationResult:
        """Tune PD gains using a simplified Ziegler-Nichols procedure."""
        kp_pos_ref = np.array(kp_pos_ref) if kp_pos_ref is not None else np.array([10.0, 10.0, 15.0])
        kd_pos_ref = np.array(kd_pos_ref) if kd_pos_ref is not None else np.array([5.0, 5.0, 8.0])
        kp_angle_ref = np.array(kp_angle_ref) if kp_angle_ref is not None else np.array([100.0, 100.0, 100.0])
        kd_angle_ref = np.array(kd_angle_ref) if kd_angle_ref is not None else np.array([10.0, 10.0, 10.0])

        self.evaluation_count = 0
        self.history = []
        self.progress_callback = progress_callback

        ku_factor, tu = self._estimate_ultimate_gain(
            kp_pos_ref,
            kd_pos_ref,
            kp_angle_ref,
            kd_angle_ref,
            max_steps=max_iterations,
        )

        ku_pos = kp_pos_ref * ku_factor
        ku_angle = kp_angle_ref * ku_factor

        optimized_kp_pos = 0.8 * ku_pos
        optimized_kd_pos = ku_pos * tu / 8.0
        optimized_kp_angle = kp_angle_ref.copy()
        optimized_kd_angle = kd_angle_ref.copy()

        best_metrics = self._run_simulation_with_gains(
            optimized_kp_pos,
            optimized_kd_pos,
            optimized_kp_angle,
            optimized_kd_angle,
        )

        objective = (
            3.0 * best_metrics["rmse_pos"] +
            1.0 * best_metrics["rmse_vel"] +
            2.0 * np.radians(best_metrics["rmse_yaw_deg"]) +
            0.1 * best_metrics["smoothness"]
        )

        self.evaluation_count = 1
        history_entry = {
            "evaluation": 1,
            "objective": float(objective),
            "metrics": best_metrics,
            "gains": {
                "kp_pos": optimized_kp_pos.copy(),
                "kd_pos": optimized_kd_pos.copy(),
                "kp_angle": optimized_kp_angle.copy(),
                "kd_angle": optimized_kd_angle.copy(),
            }
        }
        self.history.append(history_entry)
        if self.progress_callback:
            self.progress_callback(history_entry)

        return OptimizationResult(
            best_gains={
                "kp_pos": optimized_kp_pos,
                "kd_pos": optimized_kd_pos,
                "kp_angle": optimized_kp_angle,
                "kd_angle": optimized_kd_angle,
            },
            best_metrics=best_metrics,
            optimization_history=self.history,
            num_evaluations=self.evaluation_count,
        )
