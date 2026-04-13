from nicegui import ui
import plotly.graph_objects as go
import asyncio
import numpy as np

from sim.simulator import SimulationEngine
from sim.trajectories import PATHS, build_generator, trajectory_fn_from_generator
from sim.visualization import (
    make_3d_trajectory_plot,
    make_position_plot,
    make_rpy_plot,
    make_velocity_plot,
    make_initial_3d_figure,
    make_incremental_3d_figure,
    update_incremental_3d_figure,
    make_angular_velocity_plot,
    make_energy_plot,
    make_acceleration_plot,
    make_jerk_plot,
    make_snap_plot,
)

METHODS = {
    "Smooth Only": "smooth",
    "Minimum Jerk": "jerk",
    "Minimum Snap": "snap",
}

CONTROLLERS = {
    "PD (Stable)": "pd",
    "Geometric (Mellinger-style)": "geometric",
}


ui.add_head_html('''
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0e1117;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #31333f;
    }
</style>
''', shared=True)


class State:
    def __init__(self):
        self.path_name = "Path 3"
        self.method = "snap"
        self.controller_mode = "pd"
        self.fnoise = 1.0


state = State()


def _rms_xyz(xyz: np.ndarray) -> float:
    if xyz.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(xyz ** 2)))


def _theoretical_profiles(path_name: str, method: str, n_samples: int = 1200):
    generator = build_generator(path_name, method)
    t_end = float(getattr(generator, "total_duration", 25.0))
    times = np.linspace(0.0, t_end, n_samples)

    acc = np.array([generator.evaluate(float(t))[6:9] for t in times])

    if len(times) < 2:
        jerk = np.zeros_like(acc)
        snap = np.zeros_like(acc)
    else:
        dt = float(times[1] - times[0])
        jerk = np.gradient(acc, dt, axis=0)
        snap = np.gradient(jerk, dt, axis=0)

    return {
        "times": times,
        "acc": acc,
        "jerk": jerk,
        "snap": snap,
        "acc_norm": np.linalg.norm(acc, axis=1),
        "jerk_norm": np.linalg.norm(jerk, axis=1),
        "snap_norm": np.linalg.norm(snap, axis=1),
        "acc_rms": _rms_xyz(acc),
        "jerk_rms": _rms_xyz(jerk),
        "snap_rms": _rms_xyz(snap),
    }


@ui.page('/')
def main_page():
    ui.markdown('## HKUST ELEC5660 — Quadrotor Simulator').classes('text-2xl font-bold mb-4')

    with ui.row().classes('w-full items-start flex-nowrap gap-4'):
        with ui.card().classes('w-80 shrink-0 p-4 gap-4'):
            ui.label('Simulation Settings').classes('text-lg font-bold')

            ui.select(
                list(PATHS.keys()),
                value=state.path_name,
                label='Path',
                on_change=lambda e: setattr(state, 'path_name', e.value),
            ).classes('w-full')

            default_method = next(label for label, key in METHODS.items() if key == state.method)
            ui.select(
                list(METHODS.keys()),
                value=default_method,
                label='Optimization',
                on_change=lambda e: setattr(state, 'method', METHODS[e.value]),
            ).classes('w-full mt-2')

            default_ctrl = next(label for label, key in CONTROLLERS.items() if key == state.controller_mode)
            ui.select(
                list(CONTROLLERS.keys()),
                value=default_ctrl,
                label='Controller',
                on_change=lambda e: setattr(state, 'controller_mode', CONTROLLERS[e.value]),
            ).classes('w-full mt-2')

            ui.label('Disturbance std (N)').classes('mt-2')
            slider_label = ui.label(f'{state.fnoise:.1f}')

            def update_noise(e):
                val = float(e.value)
                state.fnoise = val
                slider_label.set_text(f'{val:.1f}')

            ui.slider(min=0.0, max=3.0, step=0.1, value=state.fnoise, on_change=update_noise).classes('w-full')

            run_btn = ui.button('Run simulation', on_click=lambda: start_simulation()).classes('w-full mt-4 bg-blue-600')
            compare_btn = ui.button('Compare all methods', on_click=lambda: compare_methods()).classes('w-full mt-2 bg-teal-600')
            compare_ctrl_btn = ui.button('Compare controllers (A/B)', on_click=lambda: compare_controllers()).classes('w-full mt-2 bg-amber-600')

            ui.label('Last Run Metrics').classes('text-md font-bold mt-6')
            metrics_container = ui.column().classes('w-full gap-2')

            ui.label('Method Comparison').classes('text-md font-bold mt-6')
            compare_columns = [
                {'name': 'method', 'label': 'Method', 'field': 'method'},
                {'name': 'acc_rms', 'label': 'Acc RMS', 'field': 'acc_rms'},
                {'name': 'jerk_rms', 'label': 'Jerk RMS', 'field': 'jerk_rms'},
                {'name': 'snap_rms', 'label': 'Snap RMS', 'field': 'snap_rms'},
                {'name': 'energy', 'label': 'Energy', 'field': 'energy'},
                {'name': 'ang_vel', 'label': 'AngVel RMS', 'field': 'ang_vel'},
            ]
            compare_table = ui.table(columns=compare_columns, rows=[]).classes('w-full')

            ui.label('Controller A/B Comparison').classes('text-md font-bold mt-6')
            compare_ctrl_columns = [
                {'name': 'controller', 'label': 'Controller', 'field': 'controller'},
                {'name': 'rmse_pos', 'label': 'RMSE Pos', 'field': 'rmse_pos'},
                {'name': 'rmse_vel', 'label': 'RMSE Vel', 'field': 'rmse_vel'},
                {'name': 'rmse_yaw', 'label': 'RMSE Yaw', 'field': 'rmse_yaw'},
                {'name': 'energy', 'label': 'Energy', 'field': 'energy'},
                {'name': 'acc_rms', 'label': 'Acc RMS', 'field': 'acc_rms'},
                {'name': 'jerk_rms', 'label': 'Jerk RMS', 'field': 'jerk_rms'},
                {'name': 'snap_rms', 'label': 'Snap RMS', 'field': 'snap_rms'},
            ]
            compare_ctrl_table = ui.table(columns=compare_ctrl_columns, rows=[]).classes('w-full')

            ui.label('Theoretical (No Controller)').classes('text-md font-bold mt-6')
            th_btn = ui.button('Analyze theoretical profiles', on_click=lambda: analyze_theoretical()).classes('w-full mt-2 bg-indigo-600')
            th_columns = [
                {'name': 'method', 'label': 'Method', 'field': 'method'},
                {'name': 'acc_rms', 'label': 'Acc RMS', 'field': 'acc_rms'},
                {'name': 'jerk_rms', 'label': 'Jerk RMS', 'field': 'jerk_rms'},
                {'name': 'snap_rms', 'label': 'Snap RMS', 'field': 'snap_rms'},
            ]
            th_table = ui.table(columns=th_columns, rows=[]).classes('w-full')

        with ui.card().classes('grow min-w-[400px] p-4'):
            ui.label('3D Trajectory').classes('text-xl font-bold mb-2')
            fig3d = make_initial_3d_figure()
            plot3d = ui.plotly(fig3d).classes('w-full h-[600px]')

    ui.label('Analysis').classes('text-xl font-bold mt-8 mb-4')
    with ui.column().classes('w-full gap-4'):
        with ui.card().classes('w-full p-2'):
            ui.label('Attitude').classes('font-bold')
            plot_rpy = ui.plotly(go.Figure()).classes('w-full h-[400px]')

        with ui.card().classes('w-full p-2'):
            ui.label('Velocity').classes('font-bold')
            plot_vel = ui.plotly(go.Figure()).classes('w-full h-[400px]')

        with ui.card().classes('w-full p-2'):
            ui.label('Position').classes('font-bold')
            plot_pos = ui.plotly(go.Figure()).classes('w-full h-[400px]')

        with ui.card().classes('w-full p-2'):
            ui.label('Control Effort (Thrust^2)').classes('font-bold')
            plot_energy = ui.plotly(go.Figure()).classes('w-full h-[400px]')

        with ui.card().classes('w-full p-2'):
            ui.label('Angular Velocity (p, q, r)').classes('font-bold')
            plot_ang_vel = ui.plotly(go.Figure()).classes('w-full h-[400px]')

        with ui.card().classes('w-full p-2'):
            ui.label('Acceleration').classes('font-bold')
            plot_acc = ui.plotly(go.Figure()).classes('w-full h-[400px]')

        with ui.card().classes('w-full p-2'):
            ui.label('Jerk').classes('font-bold')
            plot_jerk = ui.plotly(go.Figure()).classes('w-full h-[400px]')

        with ui.card().classes('w-full p-2'):
            ui.label('Snap').classes('font-bold')
            plot_snap = ui.plotly(go.Figure()).classes('w-full h-[400px]')

        with ui.card().classes('w-full p-2'):
            ui.label('Theoretical Acc Norm Comparison').classes('font-bold')
            plot_theory_acc = ui.plotly(go.Figure()).classes('w-full h-[400px]')

        with ui.card().classes('w-full p-2'):
            ui.label('Theoretical Jerk Norm Comparison').classes('font-bold')
            plot_theory_jerk = ui.plotly(go.Figure()).classes('w-full h-[400px]')

        with ui.card().classes('w-full p-2'):
            ui.label('Theoretical Snap Norm Comparison').classes('font-bold')
            plot_theory_snap = ui.plotly(go.Figure()).classes('w-full h-[400px]')

    async def start_simulation():
        run_btn.disable()
        compare_btn.disable()
        compare_ctrl_btn.disable()
        th_btn.disable()
        run_btn.text = 'Running...'

        generator = build_generator(state.path_name, state.method)
        trajectory_fn = trajectory_fn_from_generator(generator)
        engine = SimulationEngine(
            trajectory_fn=trajectory_fn,
            controller_mode=state.controller_mode,
            t_final=25.0,
            t_step=0.002,
            control_step=0.01,
            fnoise=float(state.fnoise),
            seed=None,
        )

        metrics_container.clear()
        render_steps = 100
        current_fig, num_static_traces = make_incremental_3d_figure()

        while not engine.done:
            engine.step(controls=render_steps)
            result = engine.result(compute_metrics=False)

            if len(result.times) > 0:
                pos = result.states[:, 0:3]
                des = result.desired[:, 0:3]
                idx = len(result.times) - 1
                update_incremental_3d_figure(current_fig, num_static_traces, pos, des, idx)
                plot3d.update_figure(current_fig)

            await asyncio.sleep(0.001)

        result = engine.result(compute_metrics=True)

        final_fig3d = make_3d_trajectory_plot(result, len(result.times) - 1)
        final_fig3d.layout.uirevision = 'constant'
        final_fig3d.layout.margin = dict(l=0, r=0, t=0, b=0)
        final_fig3d.layout.height = 600
        plot3d.update_figure(final_fig3d)

        plot_rpy.update_figure(make_rpy_plot(result))
        plot_vel.update_figure(make_velocity_plot(result))
        plot_pos.update_figure(make_position_plot(result))
        plot_energy.update_figure(make_energy_plot(result))
        plot_ang_vel.update_figure(make_angular_velocity_plot(result))
        plot_acc.update_figure(make_acceleration_plot(result))
        plot_jerk.update_figure(make_jerk_plot(result))
        plot_snap.update_figure(make_snap_plot(result))

        with metrics_container:
            def metric_display(label, value):
                with ui.row().classes('metric-card w-full justify-between items-center'):
                    ui.label(label).classes('metric-label')
                    ui.label(value).classes('metric-value')

            metric_display('RMSE Position', f'{result.rmse_pos:.4f} m')
            metric_display('RMSE Velocity', f'{result.rmse_vel:.4f} m/s')
            metric_display('RMSE Yaw', f'{result.rmse_yaw_deg:.3f} deg')
            metric_display('Smoothness', f'{result.smoothness:.3f}')

            if result.energy_effort is not None:
                metric_display('Energy Effort', f'{result.energy_effort:.2f} N²·s')
            if result.angular_velocity_rms is not None:
                metric_display('Ang Vel RMS', f'{result.angular_velocity_rms:.4f} rad/s')
            if result.acceleration_rms is not None:
                metric_display('Acceleration RMS', f'{result.acceleration_rms:.4f} m/s²')
            if result.jerk_rms is not None:
                metric_display('Jerk RMS', f'{result.jerk_rms:.4f} m/s³')
            if result.snap_rms is not None:
                metric_display('Snap RMS', f'{result.snap_rms:.4f} m/s⁴')

        run_btn.enable()
        compare_btn.enable()
        compare_ctrl_btn.enable()
        th_btn.enable()
        run_btn.text = 'Run simulation'

    async def compare_methods():
        run_btn.disable()
        compare_btn.disable()
        compare_ctrl_btn.disable()
        th_btn.disable()
        compare_btn.text = 'Comparing...'

        rows = []
        methods_in_order = [('smooth', 'Smooth'), ('jerk', 'Jerk'), ('snap', 'Snap')]
        for method_key, method_label in methods_in_order:
            generator = build_generator(state.path_name, method_key)
            trajectory_fn = trajectory_fn_from_generator(generator)
            engine = SimulationEngine(
                trajectory_fn=trajectory_fn,
                controller_mode=state.controller_mode,
                t_final=25.0,
                t_step=0.002,
                control_step=0.01,
                fnoise=float(state.fnoise),
                seed=0,
            )
            while not engine.done:
                engine.step(controls=200)
                await asyncio.sleep(0)

            result = engine.result(compute_metrics=True)
            rows.append(
                {
                    'method': method_label,
                    'acc_rms': f"{result.acceleration_rms:.4f}" if result.acceleration_rms is not None else '-',
                    'jerk_rms': f"{result.jerk_rms:.4f}" if result.jerk_rms is not None else '-',
                    'snap_rms': f"{result.snap_rms:.4f}" if result.snap_rms is not None else '-',
                    'energy': f"{result.energy_effort:.2f}" if result.energy_effort is not None else '-',
                    'ang_vel': f"{result.angular_velocity_rms:.4f}" if result.angular_velocity_rms is not None else '-',
                }
            )

        compare_table.rows = rows
        compare_table.update()

        run_btn.enable()
        compare_btn.enable()
        compare_ctrl_btn.enable()
        th_btn.enable()
        compare_btn.text = 'Compare all methods'

    async def compare_controllers():
        run_btn.disable()
        compare_btn.disable()
        compare_ctrl_btn.disable()
        th_btn.disable()
        compare_ctrl_btn.text = 'Comparing controllers...'

        rows = []
        generator = build_generator(state.path_name, state.method)
        trajectory_fn = trajectory_fn_from_generator(generator)

        for mode_key, mode_label in [('pd', 'PD (Stable)'), ('geometric', 'Geometric (Mellinger-style)')]:
            engine = SimulationEngine(
                trajectory_fn=trajectory_fn,
                controller_mode=mode_key,
                t_final=25.0,
                t_step=0.002,
                control_step=0.01,
                fnoise=float(state.fnoise),
                seed=0,
            )
            while not engine.done:
                engine.step(controls=200)
                await asyncio.sleep(0)

            result = engine.result(compute_metrics=True)
            rows.append(
                {
                    'controller': mode_label,
                    'rmse_pos': f"{result.rmse_pos:.4f}" if result.rmse_pos is not None else '-',
                    'rmse_vel': f"{result.rmse_vel:.4f}" if result.rmse_vel is not None else '-',
                    'rmse_yaw': f"{result.rmse_yaw_deg:.3f}" if result.rmse_yaw_deg is not None else '-',
                    'energy': f"{result.energy_effort:.2f}" if result.energy_effort is not None else '-',
                    'acc_rms': f"{result.acceleration_rms:.4f}" if result.acceleration_rms is not None else '-',
                    'jerk_rms': f"{result.jerk_rms:.4f}" if result.jerk_rms is not None else '-',
                    'snap_rms': f"{result.snap_rms:.4f}" if result.snap_rms is not None else '-',
                }
            )

        compare_ctrl_table.rows = rows
        compare_ctrl_table.update()

        run_btn.enable()
        compare_btn.enable()
        compare_ctrl_btn.enable()
        th_btn.enable()
        compare_ctrl_btn.text = 'Compare controllers (A/B)'

    async def analyze_theoretical():
        run_btn.disable()
        compare_btn.disable()
        compare_ctrl_btn.disable()
        th_btn.disable()
        th_btn.text = 'Analyzing...'

        method_defs = [('smooth', 'Smooth'), ('jerk', 'Jerk'), ('snap', 'Snap')]
        color_map = {'Smooth': '#7f7f7f', 'Jerk': '#1f77b4', 'Snap': '#d62728'}

        profiles = {}
        rows = []
        for key, label in method_defs:
            p = _theoretical_profiles(state.path_name, key)
            profiles[label] = p
            rows.append(
                {
                    'method': label,
                    'acc_rms': f"{p['acc_rms']:.4f}",
                    'jerk_rms': f"{p['jerk_rms']:.4f}",
                    'snap_rms': f"{p['snap_rms']:.4f}",
                }
            )
            await asyncio.sleep(0)

        th_table.rows = rows
        th_table.update()

        fig_acc = go.Figure()
        fig_jerk = go.Figure()
        fig_snap = go.Figure()
        for label in ['Smooth', 'Jerk', 'Snap']:
            p = profiles[label]
            color = color_map[label]
            fig_acc.add_trace(go.Scatter(x=p['times'], y=p['acc_norm'], name=label, line=dict(color=color)))
            fig_jerk.add_trace(go.Scatter(x=p['times'], y=p['jerk_norm'], name=label, line=dict(color=color)))
            fig_snap.add_trace(go.Scatter(x=p['times'], y=p['snap_norm'], name=label, line=dict(color=color)))

        fig_acc.update_layout(
            title='No-Controller Acc Norm |a|',
            xaxis_title='Time (s)',
            yaxis_title='m/s²',
            margin=dict(l=30, r=10, t=40, b=30),
            legend=dict(orientation='h'),
        )
        fig_jerk.update_layout(
            title='No-Controller Jerk Norm |j|',
            xaxis_title='Time (s)',
            yaxis_title='m/s³',
            margin=dict(l=30, r=10, t=40, b=30),
            legend=dict(orientation='h'),
        )
        fig_snap.update_layout(
            title='No-Controller Snap Norm |s|',
            xaxis_title='Time (s)',
            yaxis_title='m/s⁴',
            margin=dict(l=30, r=10, t=40, b=30),
            legend=dict(orientation='h'),
        )

        plot_theory_acc.update_figure(fig_acc)
        plot_theory_jerk.update_figure(fig_jerk)
        plot_theory_snap.update_figure(fig_snap)

        run_btn.enable()
        compare_btn.enable()
        compare_ctrl_btn.enable()
        th_btn.enable()
        th_btn.text = 'Analyze theoretical profiles'


ui.run(title='ELEC5660 Simulator', port=8080)
