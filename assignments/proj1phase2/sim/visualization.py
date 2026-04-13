from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .math_utils import wrap_to_pi
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
    yaw_idx = 15 if result.desired.shape[1] >= 16 else 9
    desired_yaw = result.desired[:, yaw_idx]
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


def make_energy_plot(result: SimulationResult) -> go.Figure:
    """시간에 따른 제어 입력(Thrust) 및 제어 노력 시계열 플롯 생성"""
    if result.forces is None:
        return go.Figure().update_layout(title="No force data available")

    # 제어 노력의 피적분 함수인 Thrust^2를 시각화합니다.
    energy_inst = result.forces**2
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=result.times, y=energy_inst, name="Thrust^2", line=dict(color="#9467bd"))
    )
    
    fig.update_layout(
        title="Control Effort (Thrust Squared)",
        xaxis_title="Time (s)",
        yaxis_title="N²",
        margin=dict(l=30, r=10, t=40, b=30),
        legend=dict(orientation="h"),
        template="plotly_white"
    )
    return fig

def make_angular_velocity_plot(result: SimulationResult) -> go.Figure:
    """기체 좌표계 기준 각속도 p, q, r 시계열 플롯 생성"""
    # states의 10, 11, 12번 인덱스는 각각 p, q, r (rad/s) 입니다.
    p_rate = result.states[:, 10]
    q_rate = result.states[:, 11]
    r_rate = result.states[:, 12]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.times, y=p_rate, name="p (roll rate)", line=dict(color="#e377c2")))
    fig.add_trace(go.Scatter(x=result.times, y=q_rate, name="q (pitch rate)", line=dict(color="#7f7f7f")))
    fig.add_trace(go.Scatter(x=result.times, y=r_rate, name="r (yaw rate)", line=dict(color="#bcbd22")))
    
    fig.update_layout(
        title="Body Angular Velocity (rad/s)",
        xaxis_title="Time (s)",
        yaxis_title="rad/s",
        margin=dict(l=30, r=10, t=40, b=30),
        legend=dict(orientation="h"),
        template="plotly_white"
    )
    return fig


def make_acceleration_plot(result: SimulationResult) -> go.Figure:
    """Create true acceleration plot from simulated velocity."""
    if len(result.times) < 2:
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(
            title="True Acceleration (m/s^2)",
            xaxis_title="Time (s)",
            yaxis_title="m/s^2",
            margin=dict(l=30, r=10, t=40, b=30),
            legend=dict(orientation="h"),
        )
        return fig

    vel = result.states[:, 3:6]
    dt = float(np.mean(np.diff(result.times)))
    if dt <= 0.0:
        acc = np.zeros_like(vel)
    else:
        acc = np.zeros_like(vel)
        acc[:, 0] = np.gradient(vel[:, 0], dt)
        acc[:, 1] = np.gradient(vel[:, 1], dt)
        acc[:, 2] = np.gradient(vel[:, 2], dt)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=result.times, y=acc[:, 0], name="Ax", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=result.times, y=acc[:, 1], name="Ay", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=result.times, y=acc[:, 2], name="Az", line=dict(color="#d62728")))
    fig.update_layout(
        title="True Acceleration (m/s^2)",
        xaxis_title="Time (s)",
        yaxis_title="m/s^2",
        margin=dict(l=30, r=10, t=40, b=30),
        legend=dict(orientation="h"),
    )
    return fig


def make_jerk_plot(result: SimulationResult) -> go.Figure:
    """Create true jerk plot from simulated acceleration."""
    if len(result.times) < 2:
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(
            title="True Jerk (m/s^3)",
            xaxis_title="Time (s)",
            yaxis_title="m/s^3",
            margin=dict(l=30, r=10, t=40, b=30),
            legend=dict(orientation="h"),
        )
        return fig

    vel = result.states[:, 3:6]
    dt = float(np.mean(np.diff(result.times)))
    if dt <= 0.0:
        jerk_x = np.zeros_like(vel[:, 0])
        jerk_y = np.zeros_like(vel[:, 1])
        jerk_z = np.zeros_like(vel[:, 2])
    else:
        acc_x = np.gradient(vel[:, 0], dt)
        acc_y = np.gradient(vel[:, 1], dt)
        acc_z = np.gradient(vel[:, 2], dt)
        jerk_x = np.gradient(acc_x, dt)
        jerk_y = np.gradient(acc_y, dt)
        jerk_z = np.gradient(acc_z, dt)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=result.times, y=jerk_x, name="Jx", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=result.times, y=jerk_y, name="Jy", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=result.times, y=jerk_z, name="Jz", line=dict(color="#d62728")))
    fig.update_layout(
        title="True Jerk (m/s^3)",
        xaxis_title="Time (s)",
        yaxis_title="m/s^3",
        margin=dict(l=30, r=10, t=40, b=30),
        legend=dict(orientation="h"),
    )
    return fig


def make_snap_plot(result: SimulationResult) -> go.Figure:
    """Create true snap plot from simulated jerk."""
    if len(result.times) < 2:
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(
            title="True Snap (m/s^4)",
            xaxis_title="Time (s)",
            yaxis_title="m/s^4",
            margin=dict(l=30, r=10, t=40, b=30),
            legend=dict(orientation="h"),
        )
        return fig

    vel = result.states[:, 3:6]
    dt = float(np.mean(np.diff(result.times)))
    if dt <= 0.0:
        snap_x = np.zeros_like(vel[:, 0])
        snap_y = np.zeros_like(vel[:, 1])
        snap_z = np.zeros_like(vel[:, 2])
    else:
        acc_x = np.gradient(vel[:, 0], dt)
        acc_y = np.gradient(vel[:, 1], dt)
        acc_z = np.gradient(vel[:, 2], dt)
        jerk_x = np.gradient(acc_x, dt)
        jerk_y = np.gradient(acc_y, dt)
        jerk_z = np.gradient(acc_z, dt)
        snap_x = np.gradient(jerk_x, dt)
        snap_y = np.gradient(jerk_y, dt)
        snap_z = np.gradient(jerk_z, dt)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=result.times, y=snap_x, name="Sx", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=result.times, y=snap_y, name="Sy", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=result.times, y=snap_z, name="Sz", line=dict(color="#d62728")))
    fig.update_layout(
        title="True Snap (m/s^4)",
        xaxis_title="Time (s)",
        yaxis_title="m/s^4",
        margin=dict(l=30, r=10, t=40, b=30),
        legend=dict(orientation="h"),
    )
    return fig
