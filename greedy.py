import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import traceback
from typing import List, Dict, Tuple
import time


class GreedyScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the Greedy scheduler with dataset"""
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

    def _create_output_directories(self):
        """Create directories for output files"""
        try:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"greedy_results_{self.timestamp}"
            self.viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.viz_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error creating directories: {str(e)}")

    def _initialize_tracking(self):
        """Initialize tracking variables"""
        self.best_schedule = None
        self.best_cost = float('inf')
        self.start_time = None
        self.current_violations = {'precedence': 0, 'resource': 0}
        self.step_history = []

    def _calculate_task_priority(self, task_id: int, eligible_tasks: List[int]) -> float:
        """Calculate priority score for a task based on multiple factors"""
        task = self.tasks[task_id]

        # Calculate critical ratio (processing time / num successors)
        num_successors = len(task.get('successors', []))
        critical_ratio = task['processing_time'] / (num_successors + 1)

        # Calculate resource utilization
        total_resource_usage = sum(task['resource_requirements'].values())
        max_resource_capacity = max(self.global_resources.values())
        resource_ratio = total_resource_usage / max_resource_capacity

        # Calculate successor impact
        successor_impact = sum(
            self.tasks[succ]['processing_time']
            for succ in task.get('successors', [])
            if succ in eligible_tasks
        )

        # Combine factors (lower score = higher priority)
        priority_score = (
                0.4 * critical_ratio +
                0.3 * resource_ratio +
                0.3 * successor_impact
        )

        return priority_score

    def _get_eligible_tasks(self, scheduled_tasks: List[int]) -> List[int]:
        """Get tasks that are eligible to be scheduled next"""
        scheduled_set = set(scheduled_tasks)
        eligible = []

        for task_id in range(self.num_tasks):
            if task_id in scheduled_set:
                continue

            # Check if all predecessors are scheduled
            predecessors_scheduled = True
            for pred_id in range(self.num_tasks):
                if task_id in self.tasks[pred_id].get('successors', []):
                    if pred_id not in scheduled_set:
                        predecessors_scheduled = False
                        break

            if predecessors_scheduled:
                eligible.append(task_id)

        return eligible

    def _is_resource_available(self, task_id: int, start_time: int, resource_usage: Dict) -> bool:
        """Check if resources are available for the task at given start time"""
        task = self.tasks[task_id]
        end_time = start_time + task['processing_time']

        for t in range(start_time, end_time):
            if t not in resource_usage:
                resource_usage[t] = {r: 0 for r in self.global_resources}

            for resource, amount in task['resource_requirements'].items():
                if (resource_usage[t][resource] + amount >
                        self.global_resources[resource]):
                    return False

        return True

    def _schedule_task(self, task_id: int, start_time: int, resource_usage: Dict):
        """Update resource usage for scheduled task"""
        task = self.tasks[task_id]
        end_time = start_time + task['processing_time']

        for t in range(start_time, end_time):
            if t not in resource_usage:
                resource_usage[t] = {r: 0 for r in self.global_resources}

            for resource, amount in task['resource_requirements'].items():
                resource_usage[t][resource] += amount

        return end_time

    def optimize(self) -> Dict:
        """Execute greedy optimization algorithm"""
        print("\nStarting greedy optimization...")
        self.start_time = time.time()

        try:
            scheduled_tasks = []
            resource_usage = {}
            task_timings = {}
            current_time = 0

            while len(scheduled_tasks) < self.num_tasks:
                eligible_tasks = self._get_eligible_tasks(scheduled_tasks)

                if not eligible_tasks:
                    break

                # Calculate priorities for eligible tasks
                task_priorities = [
                    (task_id, self._calculate_task_priority(task_id, eligible_tasks))
                    for task_id in eligible_tasks
                ]

                # Sort by priority (lower score = higher priority)
                task_priorities.sort(key=lambda x: x[1])

                # Try to schedule highest priority task
                scheduled = False
                for task_id, _ in task_priorities:
                    if self._is_resource_available(task_id, current_time, resource_usage):
                        end_time = self._schedule_task(task_id, current_time, resource_usage)
                        task_timings[task_id] = {
                            'start': current_time,
                            'end': end_time
                        }
                        scheduled_tasks.append(task_id)
                        self.step_history.append({
                            'step': len(scheduled_tasks),
                            'task': task_id,
                            'start_time': current_time,
                            'end_time': end_time
                        })
                        scheduled = True
                        break

                if not scheduled:
                    current_time += 1

            # Convert results to final schedule format
            final_schedule = []
            for task_id in scheduled_tasks:
                timing = task_timings[task_id]
                final_schedule.append({
                    'task_id': task_id,
                    'start_time': float(timing['start']),
                    'end_time': float(timing['end']),
                    'processing_time': float(self.tasks[task_id]['processing_time'])
                })

            makespan = max(task['end_time'] for task in final_schedule)
            execution_time = time.time() - self.start_time

            results = {
                'performance_metrics': {
                    'makespan': float(makespan),
                    'execution_time': float(execution_time),
                    'steps': len(self.step_history),
                    'violations': self.current_violations
                },
                'schedule': final_schedule,
                'step_history': self.step_history,
                'solution_quality': {
                    'resource_utilization': self._calculate_resource_utilization(final_schedule),
                    'critical_path_ratio': self._calculate_critical_path_ratio(final_schedule)
                }
            }

            print("\nOptimization Complete:")
            print(f"Final makespan: {makespan:.2f}")
            print(f"Total steps: {len(self.step_history)}")
            print(f"Execution time: {execution_time:.2f} seconds")

            self._save_report(results)
            self.create_visualizations(results)

            return results

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            traceback.print_exc()
            return self._prepare_error_results()

    def _calculate_resource_utilization(self, schedule: List[Dict]) -> float:
        """Calculate average resource utilization"""
        try:
            makespan = max(task['end_time'] for task in schedule)
            total_utilization = 0
            num_resources = len(self.global_resources)

            for resource in self.global_resources:
                used_capacity = 0
                available_capacity = self.global_resources[resource] * makespan

                for task in schedule:
                    task_id = task['task_id']
                    duration = task['end_time'] - task['start_time']
                    used_capacity += self.tasks[task_id]['resource_requirements'][resource] * duration

                total_utilization += used_capacity / available_capacity

            return total_utilization / num_resources

        except Exception as e:
            print(f"Error calculating resource utilization: {str(e)}")
            return 0.0

    def _calculate_critical_path_ratio(self, schedule: List[Dict]) -> float:
        """Calculate ratio of makespan to critical path length"""
        try:
            makespan = max(task['end_time'] for task in schedule)
            critical_path_length = self._calculate_critical_path()
            return makespan / critical_path_length if critical_path_length > 0 else float('inf')

        except Exception as e:
            print(f"Error calculating critical path ratio: {str(e)}")
            return float('inf')

    def _calculate_critical_path(self) -> int:
        """Calculate critical path length"""
        try:
            # Calculate earliest finish time for each task
            earliest_finish = [0] * self.num_tasks
            for i in range(self.num_tasks):
                predecessors_finish = 0
                for j in range(i):
                    if i in self.tasks[j].get('successors', []):
                        predecessors_finish = max(predecessors_finish, earliest_finish[j])
                earliest_finish[i] = predecessors_finish + self.tasks[i]['processing_time']

            return max(earliest_finish)

        except Exception as e:
            print(f"Error calculating critical path: {str(e)}")
            return 0

    def _save_report(self, result: Dict):
        """Save the analysis report"""
        try:
            report_path = os.path.join(self.output_dir, 'analysis_report.json')

            # Ensure all values are JSON serializable
            def convert_to_serializable(obj):
                if isinstance(obj, (np.int_, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                return obj

            result = convert_to_serializable(result)

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"Report saved to: {report_path}")

        except Exception as e:
            print(f"Error saving report: {str(e)}")
            traceback.print_exc()

    def _prepare_error_results(self) -> Dict:
        """Prepare error results"""
        execution_time = time.time() - self.start_time if self.start_time else 0

        return {
            'performance_metrics': {
                'makespan': float('inf'),
                'execution_time': float(execution_time),
                'steps': len(self.step_history),
                'violations': {'precedence': 0, 'resource': 0}
            },
            'schedule': [],
            'step_history': [],
            'error': 'Optimization failed'
        }

    def create_visualizations(self, results: Dict):
        """Generate all visualizations"""
        try:
            import matplotlib.pyplot as plt

            self._plot_schedule(results['schedule'], plt)
            self._plot_resource_utilization(results['schedule'], plt)
            self._plot_step_progression(results['step_history'], plt)
            print(f"Visualizations saved in: {self.viz_dir}")
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            traceback.print_exc()

    def _plot_schedule(self, schedule: List[Dict], plt):
        """Create Gantt chart of the schedule"""
        plt.figure(figsize=(15, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.global_resources)))
        resource_colors = dict(zip(self.global_resources.keys(), colors))

        for task in schedule:
            task_id = task['task_id']
            resource_usage = self.tasks[task_id]['resource_requirements']
            main_resource = max(resource_usage.items(), key=lambda x: x[1])[0]

            plt.barh(y=task_id,
                     width=task['processing_time'],
                     left=task['start_time'],
                     color=resource_colors[main_resource],
                     alpha=0.6)

        legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6)
                          for color in resource_colors.values()]
        plt.legend(legend_patches, list(self.global_resources.keys()),
                   title='Main Resource', loc='center left', bbox_to_anchor=(1, 0.5))

        plt.title('Schedule (Gantt Chart)')
        plt.xlabel('Time')
        plt.ylabel('Task ID')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'schedule.png'), bbox_inches='tight')
        plt.close()

    def _plot_resource_utilization(self, schedule: List[Dict], plt):
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

        times = list(range(int(makespan) + 1))
        for resource in self.global_resources:
            usage = [timeline[t][resource] for t in times]
            plt.plot(times, usage, label=f'{resource} Usage', alpha=0.7)
            plt.axhline(y=self.global_resources[resource],
                        color='red', linestyle='--', alpha=0.3,
                        label=f'{resource} Capacity')

        plt.title('Resource Utilization Over Time')
        plt.xlabel('Time')
        plt.ylabel('Resource Usage')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'resource_utilization.png'))
        plt.close()

    def _plot_step_progression(self, step_history: List[Dict], plt):
        """Plot step-by-step progression of the schedule"""
        plt.figure(figsize=(15, 6))

        steps = [step['step'] for step in step_history]
        times = [step['end_time'] for step in step_history]

        plt.plot(steps, times, 'b-', marker='o')
        plt.title('Schedule Build-up Progress')
        plt.xlabel('Number of Scheduled Tasks')
        plt.ylabel('Completion Time')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'step_progression.png'))
        plt.close()

def main():
    """Main execution function"""
    try:
        # Choose dataset size (30, 60, 90, or 120)
        dataset_size = "60"
        json_dir = os.path.join('processed_data', f'j{dataset_size}.sm', 'json')
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        if not json_files:
            raise ValueError(f"No JSON files found in {json_dir}")

        dataset_path = os.path.join(json_dir, json_files[0])
        print(f"Using dataset: {dataset_path}")

        # Initialize and run the greedy scheduler
        scheduler = GreedyScheduler(dataset_path)
        results = scheduler.optimize()

        # Print detailed results
        print("\nDetailed Results:")
        print(f"Makespan: {results['performance_metrics']['makespan']:.2f}")
        print(f"Execution Time: {results['performance_metrics']['execution_time']:.2f} seconds")
        print(f"Steps Taken: {results['performance_metrics']['steps']}")
        print(f"Resource Utilization: {results['solution_quality']['resource_utilization']:.2%}")
        print(f"Critical Path Ratio: {results['solution_quality']['critical_path_ratio']:.2f}")
        print(f"Precedence Violations: {results['performance_metrics']['violations']['precedence']}")
        print(f"Resource Violations: {results['performance_metrics']['violations']['resource']}")
        print(f"\nResults and visualizations saved in: {scheduler.output_dir}")

        # Print additional solution insights
        print("\nSolution Insights:")
        if results['solution_quality']['critical_path_ratio'] > 1.5:
            print("- Schedule length significantly exceeds critical path length")
            print("- Consider improving resource allocation strategy")

        if results['solution_quality']['resource_utilization'] < 0.5:
            print("- Low resource utilization detected")
            print("- Potential for schedule optimization")

        print("\nVisualization files generated:")
        print("- Gantt chart (schedule.png)")
        print("- Resource utilization over time (resource_utilization.png)")
        print("- Build-up progress (step_progression.png)")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()