from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# [ADDED] Imports for saving simulation results with gain values
from pathlib import Path
from datetime import datetime

from .math_utils import quaternion_to_R, wrap_to_pi
from .simulator import SimulationResult


def _add_cube(fig: go.Figure, origin: np.ndarray, size: float, color: str, opacity: float) -> None:
    """Add a 3D cube mesh to a plotly figure."""
    x0, y0, z0 = origin
    dx = dy = dz = size
    vertices = np.array(
        [
            [x0, y0, z0],
            [x0 + dx, y0, z0],
            [x0 + dx, y0 + dy, z0],
            [x0, y0 + dy, z0],
            [x0, y0, z0 + dz],
            [x0 + dx, y0, z0 + dz],
            [x0 + dx, y0 + dy, z0 + dz],
            [x0, y0 + dy, z0 + dz],
        ]
    )
    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 6, 3, 7]
    j = [1, 2, 3, 2, 5, 3, 5, 6, 6, 7, 7, 4]
    k = [2, 3, 1, 5, 6, 6, 6, 7, 4, 4, 4, 6]
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            name="Obstacle",
            hoverinfo="skip",
            showscale=False,
        )
    )


def _add_map_traces(fig: go.Figure, map_points: np.ndarray | None) -> None:
    """Add map obstacles, start and target markers to a plotly figure."""
    if map_points is None or len(map_points) < 2:
        return
    start = map_points[0] - 0.5
    target = map_points[-1] - 0.5
    obstacles = map_points[1:-1]

    fig.add_trace(
        go.Scatter3d(
            x=[start[0]],
            y=[start[1]],
            z=[start[2]],
            mode="markers",
            name="Start",
            marker=dict(size=6, color="black"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[target[0]],
            y=[target[1]],
            z=[target[2]],
            mode="markers",
            name="Target",
            marker=dict(size=7, color="red", symbol="diamond"),
        )
    )

    for obs in obstacles:
        origin = obs - 0.9
        _add_cube(fig, origin, 0.8, "#999999", 0.45)


def _add_path_traces(fig: go.Figure, path_points: np.ndarray | None) -> None:
    """Add A* path line to a plotly figure."""
    if path_points is None or len(path_points) == 0:
        return
    fig.add_trace(
        go.Scatter3d(
            x=path_points[:, 0],
            y=path_points[:, 1],
            z=path_points[:, 2],
            mode="lines+markers",
            name="A* Path",
            line=dict(color="#ff7f0e", width=4),
            marker=dict(size=4, color="#ff7f0e"),
        )
    )


def make_initial_3d_figure(
    map_points: np.ndarray | None = None,
    path_points: np.ndarray | None = None,
) -> go.Figure:
    """Create initial empty 3D figure with map and path (if provided)."""
    fig = go.Figure()
    _add_map_traces(fig, map_points)
    _add_path_traces(fig, path_points)
    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        legend=dict(orientation="h"),
    )
    return fig


def make_incremental_3d_figure(
    map_points: np.ndarray | None = None,
    path_points: np.ndarray | None = None,
) -> tuple[go.Figure, int]:
    """
    Create 3D figure for incremental updates with static map/path and dynamic trajectory traces.
    
    Returns:
        tuple: (figure, num_static_traces) where num_static_traces is the count of 
               non-trajectory traces that come before the True/Desired/Current traces.
    """
    fig = go.Figure()
    
    # Add static map and path traces
    _add_map_traces(fig, map_points)
    _add_path_traces(fig, path_points)
    num_static_traces = len(fig.data)
    
    # Add dynamic trajectory traces
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode="lines",
        name="True",
        line=dict(color="#1f77b4", width=4),
    ))
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode="lines",
        name="Desired",
        line=dict(color="#2ca02c", width=4, dash="dash"),
    ))
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode="markers",
        name="Current",
        marker=dict(size=6, color="#d62728"),
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        legend=dict(orientation="h"),
        uirevision='constant',
    )
    
    return fig, num_static_traces


def update_incremental_3d_figure(
    fig: go.Figure,
    num_static_traces: int,
    pos: np.ndarray,
    des: np.ndarray,
    current_idx: int,
) -> None:
    """Update trajectory traces in an incremental 3D figure (in-place)."""
    fig.data[num_static_traces].x = pos[:, 0]
    fig.data[num_static_traces].y = pos[:, 1]
    fig.data[num_static_traces].z = pos[:, 2]
    
    fig.data[num_static_traces + 1].x = des[:, 0]
    fig.data[num_static_traces + 1].y = des[:, 1]
    fig.data[num_static_traces + 1].z = des[:, 2]
    
    fig.data[num_static_traces + 2].x = [pos[current_idx, 0]]
    fig.data[num_static_traces + 2].y = [pos[current_idx, 1]]
    fig.data[num_static_traces + 2].z = [pos[current_idx, 2]]


def make_3d_trajectory_plot(
    result: SimulationResult,
    index: int,
    map_points: np.ndarray | None = None,
    path_points: np.ndarray | None = None,
) -> go.Figure:
    """Create complete 3D trajectory plot with all history."""
    pos = result.states[:, 0:3]
    des = result.desired[:, 0:3]
    
    if len(result.times) == 0:
        return make_initial_3d_figure(map_points, path_points)

    idx = int(np.clip(index, 0, len(result.times) - 1))

    fig = go.Figure()
    _add_map_traces(fig, map_points)
    _add_path_traces(fig, path_points)
    
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines",
            name="True",
            line=dict(color="#1f77b4", width=4),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=des[:, 0],
            y=des[:, 1],
            z=des[:, 2],
            mode="lines",
            name="Desired",
            line=dict(color="#2ca02c", width=4, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[pos[idx, 0]],
            y=[pos[idx, 1]],
            z=[pos[idx, 2]],
            mode="markers",
            name="Current",
            marker=dict(size=6, color="#d62728"),
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h"),
    )
    return fig


def make_rpy_plot(result: SimulationResult, g: float = 9.81) -> go.Figure:
    """Create attitude (roll/pitch/yaw) plot."""
    rpy_deg = np.degrees(result.rpy)
    if len(result.times) == 0:
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(
            title="Attitude (deg)",
            xaxis_title="Time (s)",
            yaxis_title="Degrees",
            margin=dict(l=30, r=10, t=40, b=30),
            legend=dict(orientation="h"),
        )
        return fig

    desired_acc = result.desired[:, 6:9]
    desired_yaw = result.desired[:, 9]
    psi = result.rpy[:, 2]
    phi_des = (desired_acc[:, 0] * np.sin(psi) - desired_acc[:, 1] * np.cos(psi)) / g
    theta_des = (desired_acc[:, 0] * np.cos(psi) + desired_acc[:, 1] * np.sin(psi)) / g
    rpy_des = np.vstack([phi_des, theta_des, desired_yaw]).T
    rpy_des = wrap_to_pi(rpy_des)
    rpy_des_deg = np.degrees(rpy_des)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=result.times, y=rpy_deg[:, 0], name="Roll", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=result.times, y=rpy_deg[:, 1], name="Pitch", line=dict(color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=result.times, y=rpy_deg[:, 2], name="Yaw", line=dict(color="#9467bd")))
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=rpy_des_deg[:, 0],
            name="Roll des",
            line=dict(color="#d62728", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=rpy_des_deg[:, 1],
            name="Pitch des",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=rpy_des_deg[:, 2],
            name="Yaw des",
            line=dict(color="#9467bd", dash="dash"),
        )
    )
    fig.update_layout(
        title="Attitude (deg)",
        xaxis_title="Time (s)",
        yaxis_title="Degrees",
        margin=dict(l=30, r=10, t=40, b=30),
        legend=dict(orientation="h"),
    )
    return fig


def make_velocity_plot(result: SimulationResult) -> go.Figure:
    """Create velocity plot."""
    vel = result.states[:, 3:6]
    des = result.desired[:, 3:6]
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=result.times, y=vel[:, 0], name="Vx", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=result.times, y=vel[:, 1], name="Vy", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=result.times, y=vel[:, 2], name="Vz", line=dict(color="#d62728")))
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=des[:, 0],
            name="Vx des",
            line=dict(color="#1f77b4", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=des[:, 1],
            name="Vy des",
            line=dict(color="#2ca02c", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=des[:, 2],
            name="Vz des",
            line=dict(color="#d62728", dash="dash"),
        )
    )
    fig.update_layout(
        title="World Velocity (m/s)",
        xaxis_title="Time (s)",
        yaxis_title="m/s",
        margin=dict(l=30, r=10, t=40, b=30),
        legend=dict(orientation="h"),
    )
    return fig


def make_position_plot(result: SimulationResult) -> go.Figure:
    """Create position plot."""
    pos = result.states[:, 0:3]
    des = result.desired[:, 0:3]
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=result.times, y=pos[:, 0], name="X", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=result.times, y=pos[:, 1], name="Y", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=result.times, y=pos[:, 2], name="Z", line=dict(color="#d62728")))
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=des[:, 0],
            name="X des",
            line=dict(color="#1f77b4", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=des[:, 1],
            name="Y des",
            line=dict(color="#2ca02c", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=des[:, 2],
            name="Z des",
            line=dict(color="#d62728", dash="dash"),
        )
    )
    fig.update_layout(
        title="World Position (m)",
        xaxis_title="Time (s)",
        yaxis_title="m",
        margin=dict(l=30, r=10, t=40, b=30),
        legend=dict(orientation="h"),
    )
    return fig


def make_control_plot(result: SimulationResult) -> go.Figure:
    """Create control input magnitude plot: total thrust, gravity-compensated thrust, and moment norm."""
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    thrust = np.zeros(len(result.times))
    moment_norm = np.zeros(len(result.times))
    if result.forces is not None:
        thrust = np.asarray(result.forces).ravel()
    if result.moments is not None:
        moment_norm = np.linalg.norm(np.asarray(result.moments), axis=1)

    mass = result.mass if result.mass is not None else 0.5
    grav = result.grav if result.grav is not None else 9.81
    gravity_thrust = mass * grav

    vertical_thrust = np.zeros(len(result.times))
    if result.states.shape[0] > 0 and result.forces is not None:
        quats = result.states[:, 6:10]
        z_axes = np.array([quaternion_to_R(q)[2, 2] for q in quats])
        vertical_thrust = thrust * z_axes

    net_vertical_force = vertical_thrust - gravity_thrust

    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=thrust,
            name="Total Thrust F",
            line=dict(color="#1f77b4"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=vertical_thrust,
            name="World-Z Thrust F_z",
            line=dict(color="#2ca02c", dash="dash"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=[gravity_thrust] * len(result.times),
            name="Gravity Thrust mg",
            line=dict(color="#9467bd", dash="dot"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=net_vertical_force,
            name="Net Vertical Force F_z - mg",
            line=dict(color="#8c564b", dash="dot"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=moment_norm,
            name="Moment |M|",
            line=dict(color="#ff7f0e"),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Control Inputs",
        xaxis_title="Time (s)",
        margin=dict(l=30, r=10, t=40, b=30),
        legend=dict(orientation="h"),
    )
    fig.update_yaxes(title_text="Thrust (N)", secondary_y=False)
    fig.update_yaxes(title_text="Moment Norm (N·m)", secondary_y=True)
    return fig


# [ADDED] Helper functions for gain visualization and saving

def _get_gains_annotation(result: SimulationResult) -> str:
    """Generate text annotation with controller gain values."""
    gains_text = "Controller Gains:\n"
    gains_text += f"Kp_pos: [{result.kp_pos[0]:.2f}, {result.kp_pos[1]:.2f}, {result.kp_pos[2]:.2f}]\n"
    gains_text += f"Kd_pos: [{result.kd_pos[0]:.2f}, {result.kd_pos[1]:.2f}, {result.kd_pos[2]:.2f}]\n"
    gains_text += f"Kp_angle: [{result.kp_angle[0]:.2f}, {result.kp_angle[1]:.2f}, {result.kp_angle[2]:.2f}]\n"
    gains_text += f"Kd_angle: [{result.kd_angle[0]:.2f}, {result.kd_angle[1]:.2f}, {result.kd_angle[2]:.2f}]"
    return gains_text


def _add_gains_annotation_to_figure(fig: go.Figure, result: SimulationResult) -> None:
    """Add controller gains annotation to figure."""
    gains_text = _get_gains_annotation(result)
    fig.add_annotation(
        text=gains_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=10, family="monospace"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        align="left",
        xanchor="left",
        yanchor="top",
    )


# [ADDED] Function to save all simulation plots with gains to HTML files

def save_simulation_results(
    result: SimulationResult,
    result_dir: str | Path = "./simulation_results",
    trajectory_name: str = "trajectory",
    fnoise: float = 1.0,
) -> dict[str, str]:
    """
    Save all simulation plots with gain values as HTML files.
    
    Args:
        result: SimulationResult containing all simulation data
        result_dir: Directory to save results
        trajectory_name: Name of the trajectory for file naming
        fnoise: Noise standard deviation for file naming
        
    Returns:
        Dictionary with save paths for each plot
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create all plots
    fig_3d = make_3d_trajectory_plot(result, len(result.times) - 1)
    fig_rpy = make_rpy_plot(result)
    fig_vel = make_velocity_plot(result)
    fig_pos = make_position_plot(result)
    fig_control = make_control_plot(result)

    # Add gains annotations to all figures
    for fig in [fig_3d, fig_rpy, fig_vel, fig_pos, fig_control]:
        _add_gains_annotation_to_figure(fig, result)

    # Save figures
    file_paths = {}

    filename_3d = f"3d_trajectory_{trajectory_name}_{fnoise:.1f}_{timestamp}.html"
    file_paths["3d_trajectory"] = str(result_dir / filename_3d)
    fig_3d.write_html(file_paths["3d_trajectory"])

    filename_rpy = f"attitude_{trajectory_name}_{fnoise:.1f}_{timestamp}.html"
    file_paths["attitude"] = str(result_dir / filename_rpy)
    fig_rpy.write_html(file_paths["attitude"])
    
    filename_vel = f"velocity_{trajectory_name}_{fnoise:.1f}_{timestamp}.html"
    file_paths["velocity"] = str(result_dir / filename_vel)
    fig_vel.write_html(file_paths["velocity"])
    
    filename_pos = f"position_{trajectory_name}_{fnoise:.1f}_{timestamp}.html"
    file_paths["position"] = str(result_dir / filename_pos)
    fig_pos.write_html(file_paths["position"])

    filename_control = f"control_inputs_{trajectory_name}_{fnoise:.1f}_{timestamp}.html"
    file_paths["control_inputs"] = str(result_dir / filename_control)
    fig_control.write_html(file_paths["control_inputs"])
    
    # Save metrics and gains to a text file
    metrics_filename = f"metrics_{trajectory_name}_{fnoise:.1f}_{timestamp}.txt"
    metrics_path = result_dir / metrics_filename
    
    with open(metrics_path, 'w') as f:
        f.write("SIMULATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Trajectory: {trajectory_name}\n")
        f.write(f"Disturbance (fnoise): {fnoise:.1f} N\n")
        f.write(f"Simulation Time: {result.times[-1]:.3f} s\n\n")
        
        f.write("METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"RMSE Position: {result.rmse_pos:.4f} m\n")
        f.write(f"RMSE Velocity: {result.rmse_vel:.4f} m/s\n")
        f.write(f"RMSE Yaw: {result.rmse_yaw_deg:.3f} deg\n")
        f.write(f"Smoothness: {result.smoothness:.3f}\n\n")
        
        f.write("CONTROLLER GAINS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Kp_pos (Position P):  [{result.kp_pos[0]:.2f}, {result.kp_pos[1]:.2f}, {result.kp_pos[2]:.2f}]\n")
        f.write(f"Kd_pos (Position D):  [{result.kd_pos[0]:.2f}, {result.kd_pos[1]:.2f}, {result.kd_pos[2]:.2f}]\n")
        f.write(f"Kp_angle (Angle P):   [{result.kp_angle[0]:.2f}, {result.kp_angle[1]:.2f}, {result.kp_angle[2]:.2f}]\n")
        f.write(f"Kd_angle (Angle D):   [{result.kd_angle[0]:.2f}, {result.kd_angle[1]:.2f}, {result.kd_angle[2]:.2f}]\n")
    
    file_paths["metrics"] = str(metrics_path)
    
    return file_paths
