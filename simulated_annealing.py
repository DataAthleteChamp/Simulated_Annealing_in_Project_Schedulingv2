import matplotlib.pyplot as plt
import json
import random
import time
from typing import List, Dict, Tuple
import os
from datetime import datetime
import traceback
import numpy as np
from adaptive_temperature_scheduler import AdaptiveTemperatureScheduler

class SimulatedAnnealingScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the Simulated Annealing scheduler with dataset"""
        self._load_dataset(dataset_path)
        self._initialize_tracking()
        self._create_output_directories()

    def _tune_parameters(self) -> Dict:
        """Optimized parameter tuning for faster convergence"""
        try:
            # Calculate problem characteristics
            resource_complexity = self._calculate_resource_complexity()
            critical_path_length = self._calculate_critical_path_length()

            # Simplified parameters based on problem size
            if self.num_tasks < 34:  # j30
                base_temp = 1000
                base_iterations = 2000
                cooling_rate = 0.96
            elif self.num_tasks < 64:  # j60
                base_temp = 2000
                base_iterations = 3000
                cooling_rate = 0.96
            elif self.num_tasks < 94:  # j90
                base_temp = 3000
                base_iterations = 3500
                cooling_rate = 0.97
            else:  # j120
                base_temp = 4000
                base_iterations = 4000
                cooling_rate = 0.97

            # Adjust based on complexity
            complexity_factor = 1 + resource_complexity
            initial_temp = base_temp * complexity_factor

            # Aggressive minimum temperature
            min_temp = initial_temp * 0.01

            # Dynamic neighborhood size
            #neighbors_per_temp = max(3, min(10, self.num_tasks // 5))
            neighbors_per_temp = max(5, min(10, self.num_tasks // 10))
            #neighbors_per_temp = max(5, min(15, self.num_tasks // 3))

            # Early stopping parameters
            no_improve_limit = max(100, self.num_tasks * 2)

            return {
                'initial_temperature': initial_temp,
                'min_temperature': min_temp,
                'cooling_rate': cooling_rate,
                'max_iterations': base_iterations,
                'neighbors_per_temp': neighbors_per_temp,
                'no_improve_limit': no_improve_limit,
                'diversity_threshold': 0.3
            }

        except Exception as e:
            print(f"Error in parameter tuning: {str(e)}")
            return self._get_default_parameters()

    def _get_default_parameters(self) -> Dict:
        """Provide safe default parameters if tuning fails"""
        base_temp = 2000
        return {
            'initial_temperature': base_temp,
            'min_temperature': base_temp * 0.001,
            'cooling_rate': 0.95,
            'max_iterations': 5000,
            'max_stages': 4,
            'neighbors_per_temp': 10,
            'reheat_factor': 0.7,
            'plateau_threshold': 100,
            'diversity_threshold': 0.3,
            'acceptance_threshold': 0.1
        }

    def _calculate_network_density(self) -> float:
        """Calculate network density considering precedence relationships"""
        try:
            edges = 0
            potential_edges = self.num_tasks * (self.num_tasks - 1) / 2

            # Build adjacency matrix for transitive closure
            adj_matrix = np.zeros((self.num_tasks, self.num_tasks))
            for i, task in enumerate(self.tasks):
                for succ in task.get('successors', []):
                    adj_matrix[i][succ] = 1

            # Floyd-Warshall for transitive closure
            tc_matrix = adj_matrix.copy()
            for k in range(self.num_tasks):
                for i in range(self.num_tasks):
                    for j in range(self.num_tasks):
                        tc_matrix[i][j] = min(1, tc_matrix[i][j] + tc_matrix[i][k] * tc_matrix[k][j])

            edges = np.sum(tc_matrix)
            return edges / potential_edges if potential_edges > 0 else 0

        except Exception as e:
            print(f"Error calculating network density: {str(e)}")
            return 0.5

    def _calculate_critical_path_length(self) -> int:
        """Calculate critical path length using forward and backward pass"""
        try:
            # Forward pass
            early_start = [0] * self.num_tasks
            early_finish = [0] * self.num_tasks

            for i in range(self.num_tasks):
                max_pred_finish = 0
                for j in range(i):
                    if i in self.tasks[j].get('successors', []):
                        max_pred_finish = max(max_pred_finish, early_finish[j])
                early_start[i] = max_pred_finish
                early_finish[i] = early_start[i] + self.tasks[i]['processing_time']

            # Backward pass
            late_start = [max(early_finish)] * self.num_tasks
            late_finish = [max(early_finish)] * self.num_tasks

            for i in range(self.num_tasks - 1, -1, -1):
                min_succ_start = late_finish[i]
                for succ in self.tasks[i].get('successors', []):
                    min_succ_start = min(min_succ_start, late_start[succ])
                late_finish[i] = min_succ_start
                late_start[i] = late_finish[i] - self.tasks[i]['processing_time']

            # Calculate critical path length (longest path)
            return max(early_finish)

        except Exception as e:
            print(f"Error calculating critical path: {str(e)}")
            return self.num_tasks

    def _calculate_resource_complexity(self) -> float:
        """Calculate resource complexity considering utilization and variability"""
        try:
            total_complexity = 0
            num_resources = len(self.global_resources)

            for resource, capacity in self.global_resources.items():
                # Calculate average utilization
                total_demand = sum(task['resource_requirements'][resource]
                                   for task in self.tasks)
                avg_utilization = total_demand / (capacity * self.num_tasks)

                # Calculate demand variability
                demands = [task['resource_requirements'][resource] for task in self.tasks]
                demand_std = np.std(demands) if len(demands) > 1 else 0
                demand_variability = demand_std / capacity if capacity > 0 else 0

                # Combine metrics
                resource_complexity = (avg_utilization + demand_variability) / 2
                total_complexity += resource_complexity

            return total_complexity / num_resources if num_resources > 0 else 0

        except Exception as e:
            print(f"Error calculating resource complexity: {str(e)}")
            return 0.5

    def _generate_neighbor(self, schedule: List[int], temperature: float) -> List[int]:
        """Enhanced neighbor generation with adaptive move selection"""
        max_attempts = 100

        for attempt in range(max_attempts):
            try:
                neighbor = schedule.copy()

                # Adaptive move selection based on temperature
                if temperature > self.initial_temperature * 0.7:
                    # Higher temperature: more aggressive moves
                    moves = ['block_swap', 'block_reverse', 'multiple_swap']
                    weights = [0.4, 0.4, 0.2]
                else:
                    # Lower temperature: more conservative moves
                    moves = ['swap', 'insert', 'block_move']
                    weights = [0.5, 0.3, 0.2]

                move_type = random.choices(moves, weights=weights)[0]

                if move_type == 'block_swap':
                    # Swap two blocks of tasks
                    size = random.randint(2, min(5, len(neighbor) // 4))
                    pos1 = random.randint(0, len(neighbor) - size)
                    pos2 = random.randint(0, len(neighbor) - size)
                    neighbor[pos1:pos1 + size], neighbor[pos2:pos2 + size] = \
                        neighbor[pos2:pos2 + size], neighbor[pos1:pos1 + size].copy()

                elif move_type == 'block_reverse':
                    # Reverse a block of tasks
                    size = random.randint(3, min(6, len(neighbor) // 3))
                    start = random.randint(0, len(neighbor) - size)
                    neighbor[start:start + size] = reversed(neighbor[start:start + size])

                elif move_type == 'multiple_swap':
                    # Multiple pairwise swaps
                    swaps = random.randint(2, 4)
                    for _ in range(swaps):
                        i, j = random.sample(range(len(neighbor)), 2)
                        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                elif move_type == 'swap':
                    # Simple swap between two positions
                    i, j = random.sample(range(len(neighbor)), 2)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                elif move_type == 'insert':
                    # Remove and insert at different position
                    i = random.randint(0, len(neighbor) - 1)
                    j = random.randint(0, len(neighbor) - 1)
                    task = neighbor.pop(i)
                    neighbor.insert(j, task)

                else:  # block_move
                    # Move a block of tasks to a new position
                    size = random.randint(2, min(4, len(neighbor) // 5))
                    start = random.randint(0, len(neighbor) - size)
                    insert_point = random.randint(0, len(neighbor) - size)

                    # Skip if insert point is within the block
                    if insert_point >= start and insert_point <= start + size:
                        continue

                    block = neighbor[start:start + size]
                    del neighbor[start:start + size]
                    neighbor[insert_point:insert_point] = block

                if self._is_valid_schedule(neighbor):
                    return neighbor

            except Exception as e:
                print(f"Error in neighbor generation attempt {attempt}: {str(e)}")
                continue

        return schedule.copy()

    def _is_valid_insertion(self, schedule: List[int], from_idx: int, to_idx: int) -> bool:
        """Check if moving task maintains precedence relationships"""
        task = schedule[from_idx]
        task_data = self.tasks[task]

        # Get task positions
        positions = {t: idx for idx, t in enumerate(schedule)}

        # Check predecessors: must come before the new position
        for i in range(len(schedule)):
            if task in self.tasks[i].get('successors', []):
                if positions[i] >= to_idx:
                    return False

        # Check successors: must come after the new position
        for succ in task_data.get('successors', []):
            if positions[succ] <= to_idx:
                return False

        return True

    def _is_valid_block_move(self, schedule: List[int], start: int, size: int, new_pos: int) -> bool:
        """Check if block move maintains precedence relationships"""
        block = set(schedule[start:start + size])

        # Check internal dependencies
        for i in range(start, start + size):
            task = schedule[i]
            for succ in self.tasks[task].get('successors', []):
                if succ in block:
                    # Successor in same block - check relative positions
                    if schedule.index(succ) < i:
                        return False

        # Check external dependencies
        positions = {t: idx for idx, t in enumerate(schedule)}

        for task in block:
            # Check predecessors
            for i in range(len(schedule)):
                if task in self.tasks[i].get('successors', []):
                    if i not in block and positions[i] >= new_pos:
                        return False

            # Check successors
            for succ in self.tasks[task].get('successors', []):
                if succ not in block and positions[succ] <= new_pos:
                    return False

        return True

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

    def _apply_strong_perturbation(self, schedule: List[int]) -> List[int]:
        """Apply strong perturbation to escape local optima"""
        perturbed = schedule.copy()

        # Multiple perturbation attempts
        for _ in range(3):
            method = random.choice([
                'block_rotation',
                'sequence_shuffle',
                'critical_path_preserve'
            ])

            if method == 'block_rotation':
                # Rotate multiple blocks
                block_size = len(perturbed) // 4
                for _ in range(2):
                    start = random.randint(0, len(perturbed) - block_size)
                    block = perturbed[start:start + block_size]
                    rotation = random.randint(1, block_size - 1)
                    block = block[rotation:] + block[:rotation]
                    perturbed[start:start + block_size] = block

            elif method == 'sequence_shuffle':
                # Preserve some critical sequences, shuffle others
                preserve_ratio = 0.3
                preserve_size = int(len(perturbed) * preserve_ratio)
                preserved = perturbed[:preserve_size]
                to_shuffle = list(set(range(self.num_tasks)) - set(preserved))
                random.shuffle(to_shuffle)
                perturbed = preserved + to_shuffle

            else:  # critical_path_preserve
                # Preserve critical path tasks, shuffle others
                critical_tasks = self._identify_critical_tasks()
                non_critical = list(set(range(self.num_tasks)) - set(critical_tasks))

                # Shuffle non-critical tasks
                positions = {task: idx for idx, task in enumerate(perturbed)}
                random.shuffle(non_critical)

                # Rebuild schedule preserving critical tasks
                new_schedule = [-1] * len(perturbed)
                non_crit_idx = 0

                for i in range(len(perturbed)):
                    if perturbed[i] in critical_tasks:
                        new_schedule[i] = perturbed[i]
                    else:
                        while (non_crit_idx < len(non_critical) and
                               not self._is_valid_position(
                                   non_critical[non_crit_idx], i, new_schedule)):
                            non_crit_idx += 1
                        if non_crit_idx < len(non_critical):
                            new_schedule[i] = non_critical[non_crit_idx]
                            non_crit_idx += 1

                if -1 not in new_schedule:
                    perturbed = new_schedule

            if self._is_valid_schedule(perturbed):
                return perturbed

        return schedule.copy()

    def _identify_critical_tasks(self) -> List[int]:
        """Identify tasks on the critical path"""
        # Calculate earliest start times
        early_start = [0] * self.num_tasks
        for i in range(self.num_tasks):
            task = self.tasks[i]
            for pred in range(i):
                if i in self.tasks[pred].get('successors', []):
                    early_start[i] = max(early_start[i],
                                         early_start[pred] + self.tasks[pred]['processing_time'])

        # Calculate latest start times
        late_start = [max(early_start) + sum(t['processing_time']
                                             for t in self.tasks)] * self.num_tasks
        for i in range(self.num_tasks - 1, -1, -1):
            task = self.tasks[i]
            for succ in task.get('successors', []):
                late_start[i] = min(late_start[i],
                                    late_start[succ] - task['processing_time'])

        # Tasks with zero total float are on critical path
        critical_tasks = []
        for i in range(self.num_tasks):
            if late_start[i] - early_start[i] < 1e-6:
                critical_tasks.append(i)

        return critical_tasks

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
            self.output_dir = f"sa_results_{self.timestamp}"
            self.viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.viz_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error creating directories: {str(e)}")

    def _initialize_tracking(self):
        """Initialize tracking variables for single-stage optimization"""
        # Auto-tune SA parameters
        self.params = self._tune_parameters()

        # Set individual parameters for easy access
        self.initial_temperature = self.params['initial_temperature']  # Keep consistent with params
        self.min_temperature = self.params['min_temperature']  # Keep consistent with params
        self.cooling_rate = self.params['cooling_rate']
        self.max_iterations = self.params['max_iterations']
        self.neighbors_per_temp = self.params['neighbors_per_temp']

        # Initialize adaptive temperature scheduler
        self.temp_scheduler = AdaptiveTemperatureScheduler(
            initial_temp=self.initial_temperature,  # Match the parameter name
            min_temp=self.min_temperature,  # Match the parameter name
            target_acceptance_rate=0.3
        )

        # Initialize tracking variables
        self.best_schedule = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.temperature_history = []
        self.acceptance_rates = []
        self.start_time = None
        self.current_violations = {'precedence': 0, 'resource': 0}
        self.best_solutions = []
        self.acceptance_history = []
        self.diversity_metric = []

        # Print tuned parameters
        print("\nInitialized Parameters:")
        print(f"Initial Temperature: {self.initial_temperature:.2f}")
        print(f"Cooling Rate: {self.cooling_rate:.4f}")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"Min Temperature: {self.min_temperature:.2f}")
        print(f"Neighbors per Temperature: {self.neighbors_per_temp}")

    def _prepare_results(self, final_schedule: List[Dict]) -> Dict:
        """Prepare results without stage-related metrics"""
        execution_time = time.time() - self.start_time
        makespan = self._calculate_makespan(final_schedule)

        return {
            'performance_metrics': {
                'makespan': float(makespan),
                'best_cost': float(self.best_cost),
                'execution_time': float(execution_time),
                'iterations': len(self.cost_history),
                'improvements': len(self.best_solutions),
                'explored_solutions': len(
                    set(tuple(sol[1]) for sol in self.best_solutions)) if self.best_solutions else 0,
                'violations': self.current_violations
            },
            'schedule': final_schedule,
            'algorithm_parameters': {
                'initial_temperature': float(self.initial_temperature),  # Match parameter name
                'final_temperature': float(self.min_temperature),  # Match parameter name
                'cooling_rate': float(self.cooling_rate),
                'max_iterations': int(self.max_iterations)
            },
            'convergence_history': {
                'costs': [float(c) for c in self.cost_history],
                'temperatures': [float(t) for t in self.temperature_history],
                'acceptance_rates': [float(r) for r in self.acceptance_rates],
                'diversity': [float(d) for d in self.diversity_metric]
            }
        }

    def _prepare_error_results(self) -> Dict:
        """Prepare error results for single-stage optimization"""
        execution_time = time.time() - self.start_time if self.start_time else 0

        return {
            'performance_metrics': {
                'makespan': float('inf'),
                'best_cost': float('inf'),
                'execution_time': float(execution_time),
                'iterations': len(self.cost_history),
                'improvements': 0,
                'explored_solutions': 0,
                'violations': {'precedence': 0, 'resource': 0}
            },
            'schedule': [],
            'algorithm_parameters': {
                'initial_temperature': float(self.initial_temperature),  # Match parameter name
                'final_temperature': float(self.min_temperature),  # Match parameter name
                'cooling_rate': float(self.cooling_rate),
                'max_iterations': int(self.max_iterations)
            },
            'convergence_history': {
                'costs': [float(c) for c in self.cost_history],
                'temperatures': [float(t) for t in self.temperature_history],
                'acceptance_rates': [float(r) for r in self.acceptance_rates],
                'diversity': [float(d) for d in self.diversity_metric]
            },
            'error': 'Optimization failed'
        }

    def optimize(self) -> Dict:
        """Single-stage optimization with adaptive temperature scheduling."""
        print("\nStarting optimization process...")
        self.start_time = time.time()

        try:
            # Initialize with multiple solutions
            solutions_pool = [self._create_initial_solution() for _ in range(3)]
            costs = [self._calculate_cost(s)[0] for s in solutions_pool]

            current_schedule = solutions_pool[costs.index(min(costs))]
            current_cost = min(costs)

            self.best_schedule = current_schedule.copy()
            self.best_cost = current_cost

            print(f"Initial solution makespan: {current_cost:.2f}")

            iterations = 0
            iterations_without_improvement = 0
            total_improvements = 0
            critical_path_length = self._calculate_critical_path_length()

            # Main optimization loop
            while (iterations < self.max_iterations and
                   iterations_without_improvement < self.params['no_improve_limit']):

                improved_in_temp = False

                # Try multiple neighbors at each temperature
                for _ in range(self.neighbors_per_temp):
                    neighbor = self._generate_neighbor(current_schedule, self.temp_scheduler.current_temp)
                    neighbor_cost = self._calculate_cost(neighbor)[0]

                    # Calculate acceptance probability
                    if neighbor_cost < current_cost:
                        acceptance_prob = 1.0
                        improved_in_temp = True
                        iterations_without_improvement = 0
                        total_improvements += 1

                        if neighbor_cost < self.best_cost:
                            self.best_schedule = neighbor.copy()
                            self.best_cost = neighbor_cost
                            print(f"New best solution: {self.best_cost:.2f}")
                    else:
                        delta = (neighbor_cost - current_cost) / current_cost
                        acceptance_prob = np.exp(-delta / self.temp_scheduler.current_temp)

                    # Determine if neighbor is accepted
                    accepted = random.random() < acceptance_prob
                    if accepted:
                        current_schedule = neighbor
                        current_cost = neighbor_cost

                    # Update temperature using adaptive scheduler
                    self.temp_scheduler.get_next_temperature(
                        current_cost=current_cost,
                        best_cost=self.best_cost,
                        iteration=iterations,
                        was_accepted=accepted
                    )

                    # Update tracking metrics
                    self._update_tracking(current_cost, self.temp_scheduler.current_temp, acceptance_prob)

                if not improved_in_temp:
                    iterations_without_improvement += 1

                    # Lower the minimum temperature if no improvement for too long
                    if iterations_without_improvement > self.temp_scheduler.window_size:
                        self.temp_scheduler.min_temp *= 0.9
                        print(f"Reducing minimum temperature to {self.temp_scheduler.min_temp:.2f} due to stagnation.")

                # Progress reporting every 100 iterations
                # if iterations % 100 == 0:
                #     stats = self.temp_scheduler.get_statistics()
                #     print(f"\nProgress:")
                #     print(f"Temperature: {stats['current_temperature']:.2f}")
                #     print(f"Cooling Rate: {stats['cooling_rate']:.4f}")
                #     print(f"Acceptance Rate: {stats['recent_acceptance_rate']:.2%}")
                #     print(f"Current best: {self.best_cost:.2f}")
                #     print(f"Iterations without improvement: {iterations_without_improvement}")
                #     print(f"Total improvements: {total_improvements}")

                iterations += 1

                # Early stopping if we reach a good solution
                if self.best_cost <= critical_path_length * 1.05:
                    print("\nReached near-optimal solution. Stopping early.")
                    break

            # Prepare final results
            execution_time = time.time() - self.start_time
            final_schedule = self._calculate_final_schedule()
            results = self._prepare_results(final_schedule)

            print("\nOptimization Complete:")
            print(f"Final makespan: {self.best_cost:.2f}")
            print(f"Total iterations: {iterations}")
            print(f"Total improvements: {total_improvements}")
            print(f"Execution time: {execution_time:.2f} seconds")

            self._save_report(results)
            self.create_visualizations()

            return results

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            traceback.print_exc()
            return self._prepare_error_results()

    def _is_valid_schedule(self, schedule: List[int]) -> bool:
        """Validate schedule feasibility"""
        try:
            # Check for duplicates
            if len(set(schedule)) != self.num_tasks:
                return False

            # Check precedence constraints
            task_positions = {task_id: pos for pos, task_id in enumerate(schedule)}
            for task_id in schedule:
                task = self.tasks[task_id]
                for succ in task.get('successors', []):
                    if task_positions[succ] < task_positions[task_id]:
                        return False

            # Calculate resource usage and check constraints
            resource_usage = {}
            task_times = self._calculate_task_times(schedule)

            for task_id in schedule:
                task = self.tasks[task_id]
                start_time = task_times[task_id]['start']
                end_time = task_times[task_id]['end']

                for t in range(start_time, end_time):
                    if t not in resource_usage:
                        resource_usage[t] = {r: 0 for r in self.global_resources}

                    for resource, amount in task['resource_requirements'].items():
                        resource_usage[t][resource] += amount
                        if resource_usage[t][resource] > self.global_resources[resource]:
                            return False

            return True

        except Exception as e:
            print(f"Error validating schedule: {str(e)}")
            return False

    def _create_initial_solution(self) -> List[int]:
        """Create initial feasible solution using topological sort"""
        # Create adjacency list
        adj_list = {i: [] for i in range(self.num_tasks)}
        in_degree = {i: 0 for i in range(self.num_tasks)}

        for task_id, task in enumerate(self.tasks):
            for succ in task.get('successors', []):
                adj_list[task_id].append(succ)
                in_degree[succ] += 1

        # Find tasks with no predecessors
        available = [i for i, degree in in_degree.items() if degree == 0]
        schedule = []

        # Process tasks in topological order
        while available:
            # Choose next task based on priorities
            next_task = min(available, key=lambda x: (
                self.tasks[x]['processing_time'],  # Shorter tasks first
                -sum(self.tasks[x]['resource_requirements'].values())  # Less resources first
            ))

            schedule.append(next_task)
            available.remove(next_task)

            # Add successors if all predecessors processed
            for succ in adj_list[next_task]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    available.append(succ)

        return schedule

    def _calculate_task_times(self, schedule: List[int]) -> Dict:
        """Calculate task timings considering resource constraints"""
        task_times = {}
        resource_usage = {}

        for task_id in schedule:
            task = self.tasks[task_id]
            start_time = 0

            # Consider dependencies
            for dep_id in schedule[:schedule.index(task_id)]:
                if task_id in self.tasks[dep_id].get('successors', []):
                    if dep_id in task_times:
                        start_time = max(start_time, task_times[dep_id]['end'])

            # Find earliest time with available resources
            while True:
                can_start = True
                end_time = start_time + task['processing_time']

                # Check resource availability
                for t in range(start_time, end_time):
                    if t not in resource_usage:
                        resource_usage[t] = {r: 0 for r in self.global_resources}

                    for resource, amount in task['resource_requirements'].items():
                        if (resource_usage[t][resource] + amount >
                                self.global_resources[resource]):
                            can_start = False
                            break
                    if not can_start:
                        break

                if can_start:
                    # Update resource usage
                    for t in range(start_time, end_time):
                        for resource, amount in task['resource_requirements'].items():
                            resource_usage[t][resource] += amount
                    break

                start_time += 1

            task_times[task_id] = {'start': start_time, 'end': end_time}

        return task_times

    def _calculate_final_schedule(self) -> List[Dict]:
        """Convert best schedule to detailed timing information"""
        try:
            if not self.best_schedule:
                return []

            schedule = []
            task_times = self._calculate_task_times(self.best_schedule)

            for task_id in self.best_schedule:
                timing = task_times[task_id]
                schedule.append({
                    'task_id': task_id,
                    'start_time': float(timing['start']),
                    'end_time': float(timing['end']),
                    'processing_time': float(self.tasks[task_id]['processing_time'])
                })

            # Sort by start time
            schedule = sorted(schedule, key=lambda x: x['start_time'])

            return schedule

        except Exception as e:
            print(f"Error calculating final schedule: {str(e)}")
            return []

    def _calculate_makespan(self, schedule: List[Dict]) -> float:
        """Calculate makespan of final schedule"""
        try:
            if not schedule:
                return 0.0
            return max(task['end_time'] for task in schedule)
        except Exception as e:
            print(f"Error calculating makespan: {str(e)}")
            return 0.0

    def _calculate_cost(self, schedule: List[int]) -> Tuple[float, Dict]:
        """Calculate cost ensuring zero violations"""
        try:
            if not self._is_valid_schedule(schedule):
                return float('inf'), {'precedence': 1, 'resource': 1}

            # Calculate task timings
            task_times = self._calculate_task_times(schedule)

            # Calculate makespan
            makespan = max(timing['end'] for timing in task_times.values())

            return float(makespan), {'precedence': 0, 'resource': 0}

        except Exception as e:
            print(f"Error calculating cost: {str(e)}")
            return float('inf'), {'precedence': 0, 'resource': 0}

    def create_visualizations(self):
        """Generate all visualizations"""
        if not self.best_schedule:
            print("No schedule to visualize")
            return

        try:
            self._plot_convergence()
            self._plot_schedule()
            self._plot_resource_utilization()
            self._plot_acceptance_rates()
            print(f"Visualizations saved in: {self.viz_dir}")
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

    def _plot_convergence(self):
        """Plot optimization convergence history"""
        plt.figure(figsize=(15, 10))

        # Cost history
        plt.subplot(2, 1, 1)
        plt.plot(self.cost_history, 'b-', label='Cost', alpha=0.7)
        plt.axhline(y=self.best_cost, color='g', linestyle='--',
                    label=f'Best Cost: {self.best_cost:.0f}')
        plt.title('Optimization Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Temperature history
        plt.subplot(2, 1, 2)
        plt.plot(self.temperature_history, 'r-', label='Temperature')
        plt.title('Temperature Cooling Schedule')
        plt.xlabel('Iteration')
        plt.ylabel('Temperature')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'convergence.png'))
        plt.close()

    def _plot_schedule(self):
        """Create Gantt chart of the schedule"""
        final_schedule = self._calculate_final_schedule()

        plt.figure(figsize=(15, 8))

        # Plot tasks
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.global_resources)))
        resource_colors = dict(zip(self.global_resources.keys(), colors))

        for task in final_schedule:
            task_id = task['task_id']
            # Color based on most used resource
            resource_usage = self.tasks[task_id]['resource_requirements']
            main_resource = max(resource_usage.items(), key=lambda x: x[1])[0]

            plt.barh(y=task_id,
                     width=task['processing_time'],
                     left=task['start_time'],
                     color=resource_colors[main_resource],
                     alpha=0.6)

        # Add resource legend
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

    def _plot_resource_utilization(self):
        """Plot resource utilization over time"""
        final_schedule = self._calculate_final_schedule()
        makespan = self._calculate_makespan(final_schedule)

        # Calculate resource usage
        timeline = {t: {r: 0 for r in self.global_resources}
                    for t in range(int(makespan) + 1)}

        for task in final_schedule:
            start = int(task['start_time'])
            end = int(task['end_time'])
            task_id = task['task_id']

            for t in range(start, end):
                for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                    timeline[t][resource] += amount

        # Create plot
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

    def _plot_acceptance_rates(self):
        """Plot acceptance rates over time"""
        plt.figure(figsize=(15, 6))

        plt.plot(self.acceptance_rates, 'g-', label='Acceptance Rate', alpha=0.7)
        plt.axhline(y=np.mean(self.acceptance_rates), color='r', linestyle='--',
                    label=f'Average: {np.mean(self.acceptance_rates):.2%}')

        plt.title('Solution Acceptance Rate Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Acceptance Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'acceptance_rates.png'))
        plt.close()

    def _is_valid_position(self, task_id: int, position: int, schedule: List[int]) -> bool:
        """Check if a task can be placed at the given position"""
        try:
            # Ignore invalid positions in schedule
            valid_schedule = [x for x in schedule if x != -1]
            task_positions = {task: pos for pos, task in enumerate(valid_schedule)}

            # Check predecessors
            for i in range(self.num_tasks):
                if task_id in self.tasks[i].get('successors', []):
                    if i in task_positions and task_positions[i] >= position:
                        return False

            # Check successors
            for succ in self.tasks[task_id].get('successors', []):
                if succ in task_positions and task_positions[succ] <= position:
                    return False

            return True
        except Exception as e:
            print(f"Error checking position validity: {str(e)}")
            return False

    def _update_tracking(self, current_cost: float, temperature: float, acceptance_prob: float):
        """Update tracking metrics during optimization"""
        try:
            self.cost_history.append(float(current_cost))
            self.temperature_history.append(float(temperature))
            self.acceptance_rates.append(float(acceptance_prob))

            # Calculate diversity metric if enough solutions
            if len(self.best_solutions) > 1:
                diversity = self._calculate_diversity(
                    [sol[1] for sol in self.best_solutions[-10:]]
                )
                self.diversity_metric.append(diversity)

        except Exception as e:
            print(f"Error updating tracking: {str(e)}")


    def _calculate_diversity(self, solutions: List[List[int]]) -> float:
        """Calculate diversity metric for a set of solutions"""
        try:
            if not solutions:
                return 0.0

            n = len(solutions)
            total_distance = 0.0

            for i in range(n):
                for j in range(i + 1, n):
                    total_distance += self._solution_distance(solutions[i], solutions[j])

            max_possible = self.num_tasks * (n * (n - 1)) / 2
            return total_distance / max_possible if max_possible > 0 else 0.0

        except Exception as e:
            print(f"Error calculating diversity: {str(e)}")
            return 0.0


    def _solution_distance(self, sol1: List[int], sol2: List[int]) -> int:
        """Calculate distance between two solutions"""
        try:
            pos1 = {task: idx for idx, task in enumerate(sol1)}
            pos2 = {task: idx for idx, task in enumerate(sol2)}

            distance = 0
            for task in range(self.num_tasks):
                distance += abs(pos1[task] - pos2[task])
            return distance

        except Exception as e:
            print(f"Error calculating solution distance: {str(e)}")
            return 0

def main():
    try:
        # Choose dataset size (30, 60, 90, or 120)
        dataset_size = "120"
        json_dir = os.path.join('processed_data', f'j{dataset_size}.sm', 'json')
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        if not json_files:
            raise ValueError(f"No JSON files found in {json_dir}")

        dataset_path = os.path.join(json_dir, json_files[0])
        print(f"Using dataset: {dataset_path}")

        scheduler = SimulatedAnnealingScheduler(dataset_path)
        results = scheduler.optimize()

        # print("\nResults:")
        # print(f"Makespan: {results['performance_metrics']['makespan']:.2f}")
        # print(f"Best Cost: {results['performance_metrics']['best_cost']:.2f}")
        # print(f"Execution Time: {results['performance_metrics']['execution_time']:.2f} seconds")
        # print(f"Precedence Violations: {results['performance_metrics']['violations']['precedence']}")
        # print(f"Resource Violations: {results['performance_metrics']['violations']['resource']}")
        # print(f"\nResults and visualizations saved in: {scheduler.output_dir}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()