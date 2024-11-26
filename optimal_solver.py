import ortools
from ortools.sat.python import cp_model
import json
import time
from typing import List, Dict
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class OptimalScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the optimal scheduler with dataset"""
        self._load_dataset(dataset_path)
        self._initialize_tracking()
        self._create_output_directories()

    def _load_dataset(self, dataset_path: str):
        """Load and validate the dataset"""
        try:
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)

            self.tasks = self.dataset['tasks']
            self.num_tasks = len(self.tasks)
            self.global_resources = self.dataset['dataset_metadata']['global_resources']

            print(f"\nLoaded dataset with {self.num_tasks} tasks and {len(self.global_resources)} resources")

        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")

    def _initialize_tracking(self):
        """Initialize tracking variables"""
        self.best_schedule = None
        self.best_cost = float('inf')
        self.start_time = None
        self.current_violations = {'deadline': 0, 'resource': 0}

    def _create_output_directories(self):
        """Create output directories"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"optimal_results_{self.timestamp}"
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)

    def optimize(self) -> Dict:
        """Run optimization using OR-Tools CP-SAT solver"""
        print("Starting optimal scheduling optimization...")
        self.start_time = time.time()

        try:
            # Create model
            model = cp_model.CpModel()

            # Create variables
            horizon = sum(task['processing_time'] for task in self.tasks)
            starts = {}  # Start time variables
            ends = {}  # End time variables
            intervals = {}  # Interval variables

            # Create task variables
            for i, task in enumerate(self.tasks):
                start = model.NewIntVar(0, horizon, f'start_{i}')
                duration = task['processing_time']
                end = model.NewIntVar(0, horizon, f'end_{i}')
                interval = model.NewIntervalVar(start, duration, end, f'interval_{i}')

                starts[i] = start
                ends[i] = end
                intervals[i] = interval

            # Add precedence constraints
            for i, task in enumerate(self.tasks):
                for successor in task.get('successors', []):
                    model.Add(ends[i] <= starts[successor])

            # Add resource constraints
            for resource, capacity in self.global_resources.items():
                demands = []
                task_intervals = []

                for i, task in enumerate(self.tasks):
                    if task['resource_requirements'].get(resource, 0) > 0:
                        demands.append(task['resource_requirements'][resource])
                        task_intervals.append(intervals[i])

                model.AddCumulative(task_intervals, demands, capacity)

            # Minimize makespan
            makespan = model.NewIntVar(0, horizon, 'makespan')
            for i in range(self.num_tasks):
                model.Add(ends[i] <= makespan)
            model.Minimize(makespan)

            # Solve
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 300  # 5 minute timeout
            status = solver.Solve(model)

            # Process results
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                # Create schedule
                schedule = []
                for i in range(self.num_tasks):
                    schedule.append({
                        'task_id': i,
                        'start_time': float(solver.Value(starts[i])),
                        'end_time': float(solver.Value(ends[i])),
                        'processing_time': float(self.tasks[i]['processing_time'])
                    })

                makespan_value = float(solver.Value(makespan))
                execution_time = time.time() - self.start_time

                results = {
                    'performance_metrics': {
                        'makespan': makespan_value,
                        'execution_time': execution_time,
                        'status': 'optimal' if status == cp_model.OPTIMAL else 'feasible',
                        'violations': {'deadline': 0, 'resource': 0}
                    },
                    'schedule': schedule,
                    'solver_statistics': {
                        'branches': solver.NumBranches(),
                        'conflicts': solver.NumConflicts(),
                        'wall_time': solver.WallTime()
                    }
                }

                # Save results and create visualizations
                self._save_report(results)
                self.create_visualizations(results)

                return results

            else:
                print("No solution found")
                return {
                    'performance_metrics': {
                        'makespan': float('inf'),
                        'execution_time': float(time.time() - self.start_time),
                        'status': 'infeasible',
                        'violations': {'deadline': 0, 'resource': 0}
                    },
                    'schedule': [],
                    'solver_statistics': {}
                }

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return {
                'performance_metrics': {
                    'makespan': float('inf'),
                    'execution_time': float(time.time() - self.start_time),
                    'status': 'error',
                    'violations': {'deadline': 0, 'resource': 0}
                },
                'schedule': [],
                'error': str(e)
            }

    def create_visualizations(self, results: Dict):
        """Generate visualizations"""
        try:
            self._plot_schedule(results['schedule'])
            self._plot_resource_utilization(results['schedule'])
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

    def _plot_schedule(self, schedule: List[Dict]):
        """Create Gantt chart"""
        plt.figure(figsize=(15, 8))

        for task in schedule:
            plt.barh(y=task['task_id'],
                     width=task['processing_time'],
                     left=task['start_time'],
                     alpha=0.6)

        plt.title('Optimal Schedule')
        plt.xlabel('Time')
        plt.ylabel('Task ID')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, 'schedule.png'))
        plt.close()

    def _plot_resource_utilization(self, schedule: List[Dict]):
        """Plot resource utilization over time"""
        makespan = max(task['end_time'] for task in schedule)
        timeline = {t: {r: 0 for r in self.global_resources}
                    for t in range(int(makespan) + 1)}

        for task in schedule:
            start = int(task['start_time'])
            end = int(task['end_time'])
            task_id = task['task_id']

            for t in range(start, end):
                for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                    timeline[t][resource] += amount

        plt.figure(figsize=(15, 8))

        for resource in self.global_resources:
            usage = [timeline[t][resource] for t in range(int(makespan) + 1)]
            plt.plot(usage, label=f'{resource} Usage')
            plt.axhline(y=self.global_resources[resource],
                        color='red',
                        linestyle='--',
                        alpha=0.3,
                        label=f'{resource} Capacity')

        plt.title('Resource Utilization Over Time')
        plt.xlabel('Time')
        plt.ylabel('Resource Usage')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.viz_dir, 'resource_utilization.png'))
        plt.close()

    def _save_report(self, results: Dict):
        """Save the analysis report"""
        report_path = os.path.join(self.output_dir, 'optimal_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    # Choose dataset size (30, 60, 90, or 120)
    dataset_size = "60"

    # Get the first .json file from j30.sm/json directory
    json_dir = os.path.join('processed_data', f'j{dataset_size}.sm', 'json')
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")

    dataset_path = os.path.join(json_dir, json_files[0])
    print(f"Using dataset: {dataset_path}")

    try:
        scheduler = OptimalScheduler(dataset_path)
        results = scheduler.optimize()

        print("\nOptimization Results:")
        print(f"Makespan: {results['performance_metrics']['makespan']}")
        print(f"Execution Time: {results['performance_metrics']['execution_time']:.2f} seconds")
        print(f"Status: {results['performance_metrics']['status']}")
        print(f"\nResults and visualizations saved in: {scheduler.output_dir}")

    except Exception as e:
        print(f"Error during execution: {str(e)}")


if __name__ == "__main__":
    main()