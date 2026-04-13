from nicegui import ui, app
import plotly.graph_objects as go
import numpy as np
import asyncio
import functools
from sim.simulator import SimulationEngine
from sim.controller import Controller
from sim.trajectories import circle_trajectory, hover_trajectory, square_trajectory
from sim.model import QuadParams
from sim.visualization import (
    make_3d_trajectory_plot,
    make_position_plot,
    make_rpy_plot,
    make_velocity_plot,
    make_control_plot,
    make_initial_3d_figure,
    make_incremental_3d_figure,
    update_incremental_3d_figure,
    # [ADDED] Import save and optimization functions
    save_simulation_results,
)
# [ADDED] Import gain optimizer
from sim.optimize import GainOptimizer

TRAJECTORIES = {
    "Hover": hover_trajectory,
    "Circle": circle_trajectory,
    "Square": square_trajectory,
}

# Styles
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

# Global state wrapper to handle UI events
class State:
    def __init__(self):
        self.traj_name = "Square"
        self.fnoise = 1.0
        # [ADDED] Store last simulation result for saving plots with gains
        self.last_result = None
        # [ADDED] Store optimized gains from optimization process
        self.optimized_gains = None
        # [ADDED] Store previous gains for comparison
        self.current_gains = None

state = State()

@ui.page('/')
def main_page():
    ui.markdown('## HKUST ELEC5660 — Quadrotor Simulator').classes('text-2xl font-bold mb-4')

    with ui.row().classes('w-full items-start flex-nowrap gap-4'):
        # Sidebar-like control panel
        with ui.card().classes('w-80 shrink-0 p-4 gap-4'):
            ui.label('Simulation Settings').classes('text-lg font-bold')

            ui.select(
                list(TRAJECTORIES.keys()),
                value=state.traj_name,
                label='Trajectory',
                on_change=lambda e: setattr(state, 'traj_name', e.value)
            ).classes('w-full')

            control_mode = ui.select(
                ['Model-based', 'no_forward', 'no_forward (Kp only)'],
                value='Model-based',
                label='Control Mode'
            ).classes('w-full')

            ui.label('Disturbance std (N)').classes('mt-2')

            # Using a label to show slider value
            slider_label = ui.label(f'{state.fnoise:.1f}')

            def update_noise(e):
                val = float(e.value)
                state.fnoise = val
                slider_label.set_text(f'{val:.1f}')

            ui.slider(min=0.0, max=3.0, step=0.1, value=state.fnoise,
                      on_change=update_noise).classes('w-full')

            with ui.row().classes('w-full gap-2'):
                run_btn = ui.button('Run simulation', on_click=lambda: start_simulation()).classes('flex-1 bg-blue-600')
                # [ADDED] Save plots button
                save_btn = ui.button('Save plots', on_click=lambda: save_plots()).classes('flex-1 bg-green-600')
                save_btn.disable()

            # [ADDED] Status message area for plot saving
            status_label = ui.label('').classes('text-sm mt-2 text-gray-600')

            # Metrics Area
            ui.label('Last Run Metrics').classes('text-md font-bold mt-6')
            metrics_container = ui.column().classes('w-full gap-2')

            # [ADDED] Gain Optimization Section
            ui.separator().classes('my-4')
            ui.label('Gain Optimization').classes('text-lg font-bold')
            ui.label('Method: Ziegler-Nichols').classes('text-sm text-gray-600')
            
            # [MODIFIED] Max iterations input (for critical parameter estimation)
            opt_iterations = ui.input(
                label="Max Iterations",
                value="15",
                validation={"Not a number": lambda x: str(x).isdigit()})
            opt_iterations.classes('w-full')
            
            # [ADDED] Status label and progress bar for optimization
            opt_status_label = ui.label('').classes('text-sm mt-2 text-blue-600')
            opt_progress_bar = ui.linear_progress(value=0).classes('w-full')
            opt_progress_bar.visible = False
            
            # [ADDED] Optimize and load gains buttons
            opt_btn = ui.button('Optimize Gains', on_click=lambda: start_optimization()).classes('w-full mt-2 bg-purple-600')
            load_opt_btn = ui.button('Load Optimized Gains', on_click=lambda: load_optimized_gains()).classes('w-full mt-1 bg-orange-600')
            load_opt_btn.disable()
            
            # [ADDED] Container for optimization results display
            opt_results_container = ui.column().classes('w-full gap-2 mt-2')

        # Main visualization area
        with ui.card().classes('grow min-w-[400px] p-4'):
            ui.label('3D Trajectory').classes('text-xl font-bold mb-2')

            # Initialize with empty figure
            fig3d = make_initial_3d_figure()
            plot3d = ui.plotly(fig3d).classes('w-full h-[600px]')

    # Additional Plots Section
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
             ui.label('Control Inputs').classes('font-bold')
             plot_controls = ui.plotly(go.Figure()).classes('w-full h-[400px]')

    async def start_simulation():
        run_btn.disable()
        save_btn.disable()
        run_btn.text = 'Running...'
        status_label.text = ''

        trajectory_fn = TRAJECTORIES[state.traj_name]
        if control_mode.value == "Model-based":
            controller_mode = "model_based"
        elif control_mode.value == "no_forward":
            controller_mode = "no_forward"
        else:
            controller_mode = "no_forward_kp_only"
        engine = SimulationEngine(
            trajectory_fn=trajectory_fn,
            t_final=25.0,
            t_step=0.002,
            control_step=0.01,
            fnoise=float(state.fnoise),
            seed=None,
            controller=Controller(QuadParams(), mode=controller_mode),
        )

        # [ADDED] Apply user-loaded optimized gains if available
        if state.current_gains is not None:
            engine.controller.Kp_pos = state.current_gains["kp_pos"].copy()
            engine.controller.Kd_pos = state.current_gains["kd_pos"].copy()
            engine.controller.Kp_angle = state.current_gains["kp_angle"].copy()
            engine.controller.Kd_angle = state.current_gains["kd_angle"].copy()
            status_label.text = 'Using loaded optimized gains'
        elif state.optimized_gains is not None:
            status_label.text = 'Optimized gains available; click Load Optimized Gains to use them'

        metrics_container.clear()

        render_steps = 100

        # Initialize figure for incremental updates
        current_fig, num_static_traces = make_incremental_3d_figure()

        while not engine.done:
            engine.step(controls=render_steps)
            result = engine.result(compute_metrics=False)

            if len(result.times) > 0:
                # Incremental update
                pos = result.states[:, 0:3]
                des = result.desired[:, 0:3]
                idx = len(result.times) - 1

                update_incremental_3d_figure(current_fig, num_static_traces, pos, des, idx)
                plot3d.update_figure(current_fig)

            # Allow UI to update
            await asyncio.sleep(0.001)

        # Final Render with all plots and metrics
        result = engine.result(compute_metrics=True)
        # [ADDED] Store result for saving plots with gains
        state.last_result = result

        final_fig3d = make_3d_trajectory_plot(result, len(result.times) - 1)
        final_fig3d.layout.uirevision = 'constant'
        final_fig3d.layout.margin = dict(l=0, r=0, t=0, b=0)
        final_fig3d.layout.height = 600
        plot3d.update_figure(final_fig3d)

        plot_rpy.update_figure(make_rpy_plot(result))
        plot_vel.update_figure(make_velocity_plot(result))
        plot_pos.update_figure(make_position_plot(result))
        plot_controls.update_figure(make_control_plot(result))

        with metrics_container:
            # Helper to display metric
            def metric_display(label, value):
                with ui.row().classes('metric-card w-full justify-between items-center'):
                    ui.label(label).classes('metric-label')
                    ui.label(value).classes('metric-value')

            # [ADDED] Show indicator when using optimized gains
            if state.optimized_gains is not None:
                is_optimized = True
                ui.label('Using Optimized Gains').classes('font-bold text-green-600 text-sm')
            
            metric_display("RMSE Position", f"{result.rmse_pos:.4f} m")
            metric_display("RMSE Velocity", f"{result.rmse_vel:.4f} m/s")
            metric_display("RMSE Yaw", f"{result.rmse_yaw_deg:.3f} deg")
            metric_display("Smoothness", f"{result.smoothness:.3f}")

        run_btn.enable()
        save_btn.enable()
        run_btn.text = 'Run simulation'

    # [ADDED] Function to save plots with gain values
    def save_plots():
        if state.last_result is None:
            status_label.text = 'No simulation results to save. Run simulation first.'
            ui.notify('No simulation results', type='warning')
            return

        try:
            save_btn.disable()
            status_label.text = 'Saving plots...'
            
            # [ADDED] Save all plots with gain annotations
            file_paths = save_simulation_results(
                state.last_result,
                result_dir="./simulation_results",
                trajectory_name=state.traj_name,
                fnoise=state.fnoise,
            )
            
            status_label.text = f'Plots saved to simulation_results/'
            ui.notify(
                f'Plots saved successfully!<br>Check the simulation_results/ folder.',
                type='positive',
                position='top'
            )
        except Exception as e:
            status_label.text = f'Error saving plots: {str(e)}'
            ui.notify(f'Error: {str(e)}', type='negative')
        finally:
            save_btn.enable()

    # [MODIFIED] Async function to run Ziegler-Nichols gain optimization
    async def start_optimization():
        opt_btn.disable()
        opt_status_label.text = 'Starting Ziegler-Nichols optimization...'
        opt_progress_bar.visible = True
        opt_progress_bar.value = 0
        opt_results_container.clear()
        
        try:
            # [MODIFIED] Use only Ziegler-Nichols method
            max_iter = int(opt_iterations.value)
            
            trajectory_fn = TRAJECTORIES[state.traj_name]
            # [ADDED] Create optimizer for current trajectory
            optimizer = GainOptimizer(
                trajectory_fn=trajectory_fn,
                t_final=25.0,
                fnoise=float(state.fnoise),
            )
            
            evaluation_count = [0]
            max_eval_estimate = max_iter  # Z-N uses single evaluation per iteration
            
            # [ADDED] Progress callback to update UI during optimization
            def progress_callback(entry):
                evaluation_count[0] += 1
                progress = min(evaluation_count[0] / max_eval_estimate, 0.99)
                opt_progress_bar.value = progress
                opt_status_label.text = (
                    f"Estimation step {entry['evaluation']}: "
                    f"Objective={entry['objective']:.4f} | "
                    f"RMSE_pos={entry['metrics']['rmse_pos']:.4f}m"
                )
            
            # [MODIFIED] Run Ziegler-Nichols optimization in background thread
            loop = asyncio.get_running_loop()
            opt_result = await loop.run_in_executor(
                None,
                functools.partial(
                    optimizer.optimize_ziegler_nichols,
                    max_iterations=max_iter,
                    progress_callback=progress_callback,
                ),
            )
            
            # [ADDED] Store optimized gains in state
            state.optimized_gains = opt_result.best_gains
            opt_progress_bar.value = 1.0
            
            # [ADDED] Display optimization results
            opt_results_container.clear()
            with opt_results_container:
                ui.label('Ziegler-Nichols Results').classes('font-bold')
                
                def result_row(label, value):
                    with ui.row().classes('w-full justify-between'):
                        ui.label(label).classes('text-sm')
                        ui.label(value).classes('text-sm font-semibold text-green-600')
                
                result_row("Evaluations", str(opt_result.num_evaluations))
                result_row("RMSE Position", f"{opt_result.best_metrics['rmse_pos']:.4f} m")
                result_row("RMSE Velocity", f"{opt_result.best_metrics['rmse_vel']:.4f} m/s")
                result_row("RMSE Yaw", f"{opt_result.best_metrics['rmse_yaw_deg']:.2f}°")
                result_row("Smoothness", f"{opt_result.best_metrics['smoothness']:.3f}")
                
                ui.label('Optimized Gains').classes('font-bold text-sm mt-3')
                gains_text = (
                    f"Kp_pos: [{opt_result.best_gains['kp_pos'][0]:.2f}, "
                    f"{opt_result.best_gains['kp_pos'][1]:.2f}, "
                    f"{opt_result.best_gains['kp_pos'][2]:.2f}]\n"
                    f"Kd_pos: [{opt_result.best_gains['kd_pos'][0]:.2f}, "
                    f"{opt_result.best_gains['kd_pos'][1]:.2f}, "
                    f"{opt_result.best_gains['kd_pos'][2]:.2f}]\n"
                    f"Kp_angle: [{opt_result.best_gains['kp_angle'][0]:.2f}, "
                    f"{opt_result.best_gains['kp_angle'][1]:.2f}, "
                    f"{opt_result.best_gains['kp_angle'][2]:.2f}]\n"
                    f"Kd_angle: [{opt_result.best_gains['kd_angle'][0]:.2f}, "
                    f"{opt_result.best_gains['kd_angle'][1]:.2f}, "
                    f"{opt_result.best_gains['kd_angle'][2]:.2f}]"
                )
                ui.label(gains_text).classes('text-xs font-mono bg-gray-100 p-2 rounded')
                
            opt_progress_bar.visible = False
            opt_btn.enable()
            load_opt_btn.enable()
            opt_status_label.text = 'Optimization complete!'
            ui.notify('Optimization finished! Click "Load Optimized Gains" to use them.', type='positive')
            
        except Exception as e:
            opt_status_label.text = f'Error: {str(e)}'
            ui.notify(f'Optimization error: {str(e)}', type='negative')
            opt_btn.enable()

    # [ADDED] Function to load optimized gains for next simulation
    def load_optimized_gains():
        if state.optimized_gains is None:
            ui.notify('No optimized gains available', type='warning')
            return

        state.current_gains = {
            "kp_pos": state.optimized_gains["kp_pos"].copy(),
            "kd_pos": state.optimized_gains["kd_pos"].copy(),
            "kp_angle": state.optimized_gains["kp_angle"].copy(),
            "kd_angle": state.optimized_gains["kd_angle"].copy(),
        }
        ui.notify(
            'Optimized gains loaded and will be used on next simulation run!',
            type='positive'
        )

ui.run(title='ELEC5660 Simulator', port=8080)
