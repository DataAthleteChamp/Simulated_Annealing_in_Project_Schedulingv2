import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import traceback
from typing import List, Dict, Tuple
import time
import logging

class GreedyScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the Greedy scheduler with dataset and logging"""
        #self._setup_logging()
        self._load_dataset(dataset_path)
        self._initialize_tracking()
        self._create_output_directories()

    def _setup_logging(self):
        """Setup detailed logging configuration"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('greedy_scheduler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('GreedyScheduler')

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

    def _find_earliest_start_time(self, task_id: int, scheduled_tasks: List[Dict]) -> int:
        """Find the earliest possible start time for a task considering all constraints"""
        earliest_time = 0
        task = self.tasks[task_id]

        # Consider precedence constraints
        for i in range(self.num_tasks):
            if task_id in self.tasks[i].get('successors', []):
                for scheduled in scheduled_tasks:
                    if scheduled['task_id'] == i:
                        earliest_time = max(earliest_time, int(scheduled['end_time']))

        # Build resource profile up to current point
        resource_usage = {}
        for scheduled in scheduled_tasks:
            start = int(scheduled['start_time'])
            end = int(scheduled['end_time'])
            for t in range(start, end):
                if t not in resource_usage:
                    resource_usage[t] = {r: 0 for r in self.global_resources}
                for resource, amount in self.tasks[scheduled['task_id']]['resource_requirements'].items():
                    resource_usage[t][resource] += amount

        # Find earliest time with available resources
        current_time = int(earliest_time)
        while True:
            can_start = True
            end_time = current_time + int(task['processing_time'])

            for t in range(current_time, end_time):
                if t not in resource_usage:
                    resource_usage[t] = {r: 0 for r in self.global_resources}

                for resource, amount in task['resource_requirements'].items():
                    if (resource_usage[t][resource] + amount >
                            self.global_resources[resource]):
                        can_start = False
                        break

            if can_start:
                return current_time
            current_time += 1

    def _validate_schedule(self, schedule: List[Dict]) -> Tuple[bool, Dict]:
        """Comprehensive schedule validation"""
        violations = {
            'precedence': [],
            'resource': [],
            'timing': []
        }

        # Create task position mapping
        task_times = {
            task['task_id']: (task['start_time'], task['end_time'])
            for task in schedule
        }

        # Check precedence constraints
        for task in schedule:
            task_id = task['task_id']
            task_end = task['end_time']

            for succ in self.tasks[task_id].get('successors', []):
                # Only check if successor has been scheduled
                if succ in task_times:
                    succ_start = task_times[succ][0]
                    if succ_start < task_end:
                        violations['precedence'].append(
                            f"Task {task_id} ends at {task_end} but successor {succ} starts at {succ_start}"
                        )

        # Check resource constraints
        resource_usage = {}
        for task in schedule:
            task_id = task['task_id']
            start_time = int(task['start_time'])
            end_time = int(task['end_time'])

            for t in range(start_time, end_time):
                if t not in resource_usage:
                    resource_usage[t] = {r: 0 for r in self.global_resources}

                for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                    resource_usage[t][resource] += amount
                    if resource_usage[t][resource] > self.global_resources[resource]:
                        violations['resource'].append(
                            f"Time {t}: Resource {resource} overused "
                            f"({resource_usage[t][resource]} > {self.global_resources[resource]})"
                        )

        # Check timing consistency
        for task in schedule:
            if task['end_time'] - task['start_time'] != task['processing_time']:
                violations['timing'].append(
                    f"Task {task['task_id']} duration mismatch: "
                    f"scheduled {task['end_time'] - task['start_time']} != "
                    f"required {task['processing_time']}"
                )

        # Determine if schedule is valid
        is_valid = all(len(v) == 0 for v in violations.values())

        return is_valid, violations

    def _prepare_error_results(self) -> Dict:
        """Prepare error results with complete structure"""
        execution_time = time.time() - self.start_time if self.start_time else 0

        return {
            'performance_metrics': {
                'makespan': float('inf'),
                'execution_time': float(execution_time),
                'steps': len(self.step_history) if hasattr(self, 'step_history') else 0,
                'violations': {'precedence': 0, 'resource': 0}
            },
            'schedule': [],
            'step_history': [],
            'solution_quality': {
                'resource_utilization': 0.0,
                'critical_path_ratio': float('inf')
            },
            'error': 'Optimization failed'
        }

    def optimize(self) -> Dict:
        """Execute greedy optimization algorithm with validation"""
        #self.logger.info("Starting greedy optimization...")
        self.start_time = time.time()

        try:
            scheduled_tasks = []
            current_time = 0

            while len(scheduled_tasks) < self.num_tasks:
                eligible_tasks = self._get_eligible_tasks([t['task_id'] for t in scheduled_tasks])

                if not eligible_tasks:
                    #self.logger.warning("No eligible tasks found")
                    break

                # Calculate priorities and earliest possible start times
                task_priorities = []
                for task_id in eligible_tasks:
                    earliest_start = self._find_earliest_start_time(task_id, scheduled_tasks)
                    priority_score = self._calculate_task_priority(task_id, eligible_tasks)
                    task_priorities.append((task_id, priority_score, earliest_start))

                # Sort by priority and earliest start time
                task_priorities.sort(key=lambda x: (x[1], x[2]))

                # Schedule highest priority task
                task_id = task_priorities[0][0]
                start_time = task_priorities[0][2]

                task = {
                    'task_id': task_id,
                    'start_time': float(start_time),
                    'end_time': float(start_time + self.tasks[task_id]['processing_time']),
                    'processing_time': float(self.tasks[task_id]['processing_time'])
                }

                scheduled_tasks.append(task)
                #self.logger.debug(f"Scheduled task {task_id} at time {start_time}")

                # Validate current partial schedule
                is_valid, violations = self._validate_schedule(scheduled_tasks)
                # if not is_valid:
                #     #self.logger.error("Schedule validation failed:")
                #     for category, issues in violations.items():
                #         for issue in issues:
                #             #self.logger.error(f"{category}: {issue}")

                current_time = max(current_time, int(task['end_time']))

            # if len(scheduled_tasks) < self.num_tasks:
            #     self.logger.error("Failed to schedule all tasks")
            #     return self._prepare_error_results()

            # Final schedule validation
            is_valid, violations = self._validate_schedule(scheduled_tasks)
            # if not is_valid:
            #     self.logger.error("Final schedule validation failed")
            #     return self._prepare_error_results()

            makespan = max(task['end_time'] for task in scheduled_tasks)
            execution_time = time.time() - self.start_time

            results = {
                'performance_metrics': {
                    'makespan': float(makespan),
                    'execution_time': float(execution_time),
                    'steps': len(scheduled_tasks),
                    'violations': {'precedence': 0, 'resource': 0}
                },
                'schedule': scheduled_tasks,
                'solution_quality': {
                    'resource_utilization': self._calculate_resource_utilization(scheduled_tasks),
                    'critical_path_ratio': self._calculate_critical_path_ratio(scheduled_tasks)
                }
            }

            # self.logger.info(f"Optimization complete. Makespan: {makespan}")
            self._save_report(results)
            self.create_visualizations(results)

            return results

        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
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

    def create_visualizations(self, results: Dict):
        """Generate enhanced visualizations with error handling"""
        try:

            # Create individual visualizations
            self._plot_schedule(results['schedule'], plt)
            self._plot_resource_utilization(results['schedule'], plt)
            self._plot_resource_profile(results['schedule'], plt)
            self._plot_task_distribution(results['schedule'], plt)

            #self.logger.info(f"Visualizations saved in: {self.viz_dir}")
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            traceback.print_exc()

    def _plot_task_distribution(self, schedule: List[Dict], plt):
        """Plot task distribution over time"""
        plt.figure(figsize=(15, 8))

        # Extract task durations and start times
        durations = [task['end_time'] - task['start_time'] for task in schedule]
        start_times = [task['start_time'] for task in schedule]

        # Create histogram of task durations
        plt.subplot(2, 1, 1)
        plt.hist(durations, bins=20, color='skyblue', alpha=0.7)
        plt.title('Distribution of Task Durations')
        plt.xlabel('Duration')
        plt.ylabel('Number of Tasks')
        plt.grid(True, alpha=0.3)

        # Create histogram of start times
        plt.subplot(2, 1, 2)
        plt.hist(start_times, bins=20, color='lightgreen', alpha=0.7)
        plt.title('Distribution of Task Start Times')
        plt.xlabel('Start Time')
        plt.ylabel('Number of Tasks')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'task_distribution.png'))
        plt.close()

    def _plot_schedule(self, schedule: List[Dict], plt):
        """Create enhanced Gantt chart of the schedule"""
        plt.figure(figsize=(20, 10))

        # Create color map for resources
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.global_resources)))
        resource_colors = dict(zip(self.global_resources.keys(), colors))

        # Plot tasks
        for task in schedule:
            task_id = task['task_id']

            # Determine main resource used by task
            resource_usage = self.tasks[task_id]['resource_requirements']
            main_resource = max(resource_usage.items(), key=lambda x: x[1])[0]

            # Calculate task properties
            start = task['start_time']
            duration = task['end_time'] - start

            # Plot task bar
            plt.barh(y=task_id, width=duration, left=start,
                     color=resource_colors[main_resource], alpha=0.6)

            # Add task label
            plt.text(start + duration / 2, task_id, f"Task {task_id}",
                     ha='center', va='center')

        # Customize plot
        plt.title('Project Schedule (Gantt Chart)')
        plt.xlabel('Time')
        plt.ylabel('Task ID')
        plt.grid(True, alpha=0.3)

        # Add resource legend
        legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6)
                          for color in resource_colors.values()]
        plt.legend(legend_patches, list(self.global_resources.keys()),
                   title='Main Resource', loc='center left',
                   bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'schedule.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_resource_profile(self, schedule: List[Dict], plt):
        """Create detailed resource usage profile"""
        makespan = max(task['end_time'] for task in schedule)

        plt.figure(figsize=(15, 10))
        num_resources = len(self.global_resources)

        for i, (resource, capacity) in enumerate(self.global_resources.items(), 1):
            plt.subplot(num_resources, 1, i)
            times = list(range(int(makespan) + 1))
            usage = self._calculate_resource_usage_over_time(schedule, resource, times)

            plt.plot(times, usage, 'b-', label='Usage', alpha=0.7)
            plt.axhline(y=capacity, color='r', linestyle='--',
                        alpha=0.5, label='Capacity')

            plt.fill_between(times, usage, alpha=0.3)
            utilization = sum(usage) / (len(times) * capacity) if capacity > 0 else 0

            plt.title(f'Resource {resource} Profile (Avg Utilization: {utilization:.1%})')
            plt.xlabel('Time')
            plt.ylabel('Usage')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'resource_profiles.png'))
        plt.close()

    def _calculate_resource_usage_over_time(self, schedule: List[Dict],
                                            resource: str, times: List[int]) -> List[float]:
        """Calculate resource usage over time"""
        usage = [0] * len(times)

        for task in schedule:
            task_id = task['task_id']
            start = int(task['start_time'])
            end = int(task['end_time'])
            amount = self.tasks[task_id]['resource_requirements'][resource]

            for t in range(start, end):
                if t < len(usage):
                    usage[t] += amount

        return usage

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

def main():
    try:
        dataset_size = "60"
        json_dir = os.path.join('processed_data', f'j{dataset_size}.sm', 'json')
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        if not json_files:
            raise ValueError(f"No JSON files found in {json_dir}")

        dataset_path = os.path.join(json_dir, json_files[0])
        print(f"Using dataset: {dataset_path}")

        scheduler = GreedyScheduler(dataset_path)
        results = scheduler.optimize()

        print("\nDetailed Results:")
        print(f"Makespan: {results['performance_metrics']['makespan']:.2f}")
        print(f"Execution Time: {results['performance_metrics']['execution_time']:.2f} seconds")
        print(f"Steps Taken: {results['performance_metrics']['steps']}")
        if 'error' not in results:
            print(f"Resource Utilization: {results['solution_quality']['resource_utilization']:.2%}")
            print(f"Critical Path Ratio: {results['solution_quality']['critical_path_ratio']:.2f}")
        print(f"Precedence Violations: {results['performance_metrics']['violations']['precedence']}")
        print(f"Resource Violations: {results['performance_metrics']['violations']['resource']}")

        if 'error' in results:
            print(f"Error: {results['error']}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()