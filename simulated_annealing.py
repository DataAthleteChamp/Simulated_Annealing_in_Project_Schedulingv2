import numpy as np
import matplotlib.pyplot as plt
import json
import random
import time
from typing import List, Dict, Tuple
import os
from datetime import datetime
import traceback


class SimulatedAnnealingScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the Simulated Annealing scheduler with dataset"""
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
            self.output_dir = f"sa_results_{self.timestamp}"
            self.viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.viz_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error creating directories: {str(e)}")

    def _initialize_tracking(self):
        """Initialize tracking variables and tune parameters"""
        # Auto-tune SA parameters
        tuned_params = self._tune_parameters()
        self.initial_temp = tuned_params['initial_temperature']
        self.min_temp = tuned_params['min_temperature']
        self.alpha = tuned_params['cooling_rate']
        self.max_iterations = tuned_params['max_iterations']
        self.perturbation_threshold = tuned_params['perturbation_threshold']  # Added this line

        # Print tuned parameters
        print("\nTuned Parameters:")
        print(f"Initial Temperature: {self.initial_temp:.2f}")
        print(f"Cooling Rate: {self.alpha:.4f}")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"Min Temperature: {self.min_temp:.2f}")

        # Initialize tracking variables
        self.best_schedule = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.temperature_history = []
        self.acceptance_rates = []
        self.start_time = None
        self.current_violations = {'precedence': 0, 'resource': 0}

    def _tune_parameters(self) -> Dict:
        """Tune SA parameters based on dataset size and characteristics"""
        try:
            # Calculate problem characteristics
            total_processing_time = sum(task['processing_time'] for task in self.tasks)
            resource_complexity = self._calculate_resource_complexity()
            successors_density = self._calculate_successors_density()

            # Base parameters by dataset size
            if self.num_tasks < 31:  # j30.sm
                base_temp = 1000
                base_alpha = 0.98
                base_iterations = 2000
                perturbation_threshold = 500
            elif self.num_tasks < 61:  # j60.sm
                base_temp = 2000
                base_alpha = 0.96
                base_iterations = 5000
                perturbation_threshold = 750
            elif self.num_tasks < 91:  # j90.sm
                base_temp = 3000
                base_alpha = 0.95
                base_iterations = 9000
                perturbation_threshold = 1000
            else:  # j120.sm
                base_temp = 5000
                base_alpha = 0.93
                base_iterations = 12000
                perturbation_threshold = 1500

            # Adjust based on problem characteristics
            complexity_factor = (1 + resource_complexity * 2) * (1 + successors_density * 3)

            # More aggressive adjustments for larger problems
            if self.num_tasks >= 90:
                complexity_factor *= 1.2

            initial_temperature = base_temp * complexity_factor
            cooling_rate = base_alpha ** (1 / np.sqrt(complexity_factor))
            max_iterations = int(base_iterations * complexity_factor)
            min_temperature = initial_temperature * 0.001  # Lower minimum temperature

            # Print tuning details
            print("\nTuned Parameters Details:")
            print(f"Dataset Size: j{self.num_tasks}")
            print(f"Resource Complexity: {resource_complexity:.3f}")
            print(f"Successors Density: {successors_density:.3f}")
            print(f"Complexity Factor: {complexity_factor:.3f}")
            print(f"Initial Temperature: {initial_temperature:.2f}")
            print(f"Cooling Rate: {cooling_rate:.4f}")
            print(f"Max Iterations: {max_iterations}")
            print(f"Perturbation Threshold: {perturbation_threshold}")

            return {
                'initial_temperature': initial_temperature,
                'min_temperature': min_temperature,
                'cooling_rate': cooling_rate,
                'max_iterations': max_iterations,
                'perturbation_threshold': perturbation_threshold
            }

        except Exception as e:
            print(f"Error tuning parameters: {str(e)}")
            return {
                'initial_temperature': 2000.0,
                'min_temperature': 0.1,
                'cooling_rate': 0.95,
                'max_iterations': 5000,
                'perturbation_threshold': 1000
            }

    def _calculate_resource_complexity(self) -> float:
        """Calculate resource complexity factor"""
        total_complexity = 0
        for resource, capacity in self.global_resources.items():
            usage = sum(task['resource_requirements'][resource] for task in self.tasks)
            total_complexity += usage / (capacity * self.num_tasks)
        return total_complexity / len(self.global_resources)

    def _calculate_successors_density(self) -> float:
        """Calculate density of successor relationships"""
        total_successors = sum(len(task.get('successors', [])) for task in self.tasks)
        max_possible = self.num_tasks * (self.num_tasks - 1) / 2
        return total_successors / max_possible if max_possible > 0 else 0

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

    def _is_resource_available(self, start_time: int, task: Dict, resource_usage: Dict) -> bool:
        """Check resource availability for task at given time"""
        end_time = start_time + task['processing_time']

        for t in range(start_time, end_time):
            if t not in resource_usage:
                resource_usage[t] = {r: 0 for r in self.global_resources}

            for resource, amount in task['resource_requirements'].items():
                current_usage = resource_usage[t].get(resource, 0)
                if current_usage + amount > self.global_resources[resource]:
                    return False

        return True

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

    def _create_topological_solution(self) -> List[int]:
        """Create solution using topological sort"""
        # Create adjacency list
        adj_list = {i: set() for i in range(self.num_tasks)}
        in_degree = {i: 0 for i in range(self.num_tasks)}

        for task_id, task in enumerate(self.tasks):
            for succ in task.get('successors', []):
                adj_list[task_id].add(succ)
                in_degree[succ] += 1

        # Find all tasks with no predecessors
        queue = [i for i in range(self.num_tasks) if in_degree[i] == 0]
        schedule = []

        while queue:
            task_id = queue.pop(0)
            schedule.append(task_id)

            # Remove edges from this task
            for succ in adj_list[task_id]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        return schedule if len(schedule) == self.num_tasks else list(range(self.num_tasks))

    def _generate_neighbor(self, schedule: List[int]) -> List[int]:
        """Generate feasible neighbor solution"""
        max_attempts = 100

        for attempt in range(max_attempts):
            try:
                neighbor = schedule.copy()

                # Choose move type with adjusted weights
                move_type = random.choices(
                    ['swap', 'insert', 'block_move'],
                    weights=[0.6, 0.3, 0.1]  # Focus more on simple moves
                )[0]

                if move_type == 'swap':
                    # Find valid swap
                    positions = list(range(len(neighbor)))
                    random.shuffle(positions)

                    for i in positions[:-1]:
                        for j in positions[i + 1:]:
                            if self._is_valid_swap(neighbor, i, j):
                                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                                return neighbor

                elif move_type == 'insert':
                    # Find valid insertion
                    for i in range(len(neighbor)):
                        task = neighbor[i]
                        temp = neighbor[:i] + neighbor[i + 1:]  # Remove task

                        # Try all possible insertion points
                        for j in range(len(temp) + 1):
                            candidate = temp[:j] + [task] + temp[j:]
                            if self._validate_precedence(candidate):
                                return candidate

                else:  # block_move
                    # Try different block sizes
                    for size in range(2, min(5, len(neighbor) // 4)):
                        for start in range(len(neighbor) - size):
                            block = neighbor[start:start + size]
                            remaining = neighbor[:start] + neighbor[start + size:]

                            # Try different insertion points
                            for insert_point in range(len(remaining) + 1):
                                candidate = (remaining[:insert_point] + block +
                                             remaining[insert_point:])
                                if self._validate_precedence(candidate):
                                    return candidate

            except Exception as e:
                print(f"Error in neighbor generation attempt {attempt}: {str(e)}")
                continue

        return schedule.copy()

    def _validate_precedence(self, schedule: List[int]) -> bool:
        """Validate precedence constraints"""
        positions = {task_id: pos for pos, task_id in enumerate(schedule)}

        for task_id, task in enumerate(self.tasks):
            if task_id in positions:  # Skip if task not in schedule
                task_pos = positions[task_id]
                for successor in task.get('successors', []):
                    if successor in positions and positions[successor] < task_pos:
                        return False
        return True

    def _is_valid_swap(self, schedule: List[int], i: int, j: int) -> bool:
        """Check if swapping positions i and j maintains precedence"""
        # Create schedule with proposed swap
        new_schedule = schedule.copy()
        new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

        return self._validate_precedence(new_schedule)

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

    def _is_valid_insertion(self, schedule: List[int], from_idx: int, to_idx: int) -> bool:
        """Check if moving task maintains precedence relationships"""
        task = self.tasks[schedule[from_idx]]

        # Check predecessors
        for i in range(to_idx):
            if schedule[from_idx] in self.tasks[schedule[i]].get('successors', []):
                return False

        # Check successors
        for succ in task.get('successors', []):
            succ_idx = schedule.index(succ)
            if to_idx >= succ_idx:
                return False

        return True

    def _is_valid_block(self, schedule: List[int], start: int, size: int) -> bool:
        """Check if block move maintains precedence relationships"""
        block = set(schedule[start:start + size])

        # Check for dependencies within block
        for task_id in block:
            task = self.tasks[task_id]
            for succ in task.get('successors', []):
                if succ in block:
                    return False

        # Check dependencies with rest of schedule
        for task_id in block:
            task = self.tasks[task_id]
            # Check if any task before block is a successor
            for i in range(start):
                if schedule[i] in task.get('successors', []):
                    return False
            # Check if any task after block is a predecessor
            for i in range(start + size, len(schedule)):
                if schedule[i] in task.get('successors', []):
                    return False

        return True

    def optimize(self) -> Dict:
        """Run optimization with intensified search"""
        print("Starting optimization process...")
        self.start_time = time.time()

        try:
            # Initialize tracking
            current_schedule = self._create_initial_solution()
            current_cost, current_violations = self._calculate_cost(current_schedule)

            self.best_schedule = current_schedule.copy()
            self.best_cost = current_cost
            self.current_violations = current_violations

            temperature = self.initial_temp
            iteration = 0
            no_improvement = 0
            moves_accepted = 0
            improvements = 0

            # Initialize multi-stage search
            stage = 1
            max_stages = 3
            min_iterations_per_stage = self.max_iterations // 2
            stage_temperature = temperature

            # Enhanced solution tracking
            solution_history = set()
            solution_history.add(tuple(current_schedule))
            best_solutions = []  # Track top solutions

            while stage <= max_stages:
                print(f"\nStarting search stage {stage}/{max_stages}")
                stage_improvements = 0
                stage_iterations = 0

                while temperature > self.min_temp and stage_iterations < min_iterations_per_stage:
                    if stage_iterations % 100 == 0:
                        print(f"\nStage {stage} - Iteration {stage_iterations}:")
                        print(f"Temperature: {temperature:.2f}")
                        print(f"Best makespan: {self.best_cost:.2f}")
                        print(f"Stage improvements: {stage_improvements}")
                        print(f"Solutions explored: {len(solution_history)}")

                    # Generate and evaluate neighbors
                    for _ in range(10):  # Try multiple neighbors per temperature
                        neighbor_schedule = self._generate_neighbor(current_schedule)
                        neighbor_tuple = tuple(neighbor_schedule)

                        if neighbor_tuple not in solution_history:
                            solution_history.add(neighbor_tuple)
                            neighbor_cost, neighbor_violations = self._calculate_cost(neighbor_schedule)

                            # Calculate acceptance probability
                            cost_diff = neighbor_cost - current_cost
                            if cost_diff < 0:  # Better solution
                                acceptance_probability = 1.0
                                improvements += 1
                                stage_improvements += 1
                                no_improvement = 0

                                if neighbor_cost < self.best_cost:
                                    self.best_schedule = neighbor_schedule.copy()
                                    self.best_cost = neighbor_cost
                                    self.current_violations = neighbor_violations.copy()
                                    best_solutions.append((neighbor_cost, neighbor_schedule.copy()))
                                    print(f"\nNew best solution found: {self.best_cost:.2f}")
                            else:
                                scaled_diff = min(500, cost_diff / (current_cost + 1))
                                acceptance_probability = np.exp(-scaled_diff / temperature)

                            # Accept or reject neighbor
                            if random.random() < acceptance_probability:
                                current_schedule = neighbor_schedule
                                current_cost = neighbor_cost
                                current_violations = neighbor_violations
                                moves_accepted += 1
                            else:
                                no_improvement += 1

                    # Apply perturbation if stuck
                    if no_improvement >= self.perturbation_threshold:
                        print(f"\nApplying perturbation in stage {stage}...")
                        perturbed_schedule = self._apply_perturbation(current_schedule)
                        perturbed_cost, perturbed_violations = self._calculate_cost(perturbed_schedule)

                        current_schedule = perturbed_schedule
                        current_cost = perturbed_cost
                        current_violations = perturbed_violations

                        temperature = stage_temperature * 0.5  # Reheat
                        no_improvement = 0
                        solution_history.clear()  # Reset solution history

                    # Track progress
                    self.cost_history.append(float(current_cost))
                    self.temperature_history.append(float(temperature))
                    self.acceptance_rates.append(moves_accepted / (iteration + 1))

                    # Update temperature
                    temperature *= self.alpha
                    iteration += 1
                    stage_iterations += 1

                # Prepare for next stage
                if stage < max_stages:
                    print(f"\nCompleting stage {stage}")
                    print(f"Stage improvements: {stage_improvements}")

                    # Reset for next stage
                    if best_solutions:
                        best_solutions.sort()
                        current_schedule = best_solutions[0][1].copy()
                        current_cost = best_solutions[0][0]

                    temperature = stage_temperature * (0.7 ** stage)  # Reduced starting temperature
                    stage_temperature = temperature
                    solution_history.clear()
                    no_improvement = 0

                stage += 1

            # Calculate final metrics and create report
            execution_time = time.time() - self.start_time
            final_schedule = self._calculate_final_schedule()
            makespan = self._calculate_makespan(final_schedule)

            results = {
                'performance_metrics': {
                    'makespan': float(makespan),
                    'best_cost': float(self.best_cost),
                    'execution_time': float(execution_time),
                    'iterations': int(iteration),
                    'total_improvements': int(improvements),
                    'solutions_explored': len(solution_history),
                    'acceptance_rate': float(moves_accepted / iteration) if iteration > 0 else 0.0,
                    'violations': self.current_violations
                },
                'schedule': final_schedule,
                'algorithm_parameters': {
                    'initial_temperature': float(self.initial_temp),
                    'final_temperature': float(temperature),
                    'cooling_rate': float(self.alpha),
                    'max_iterations': int(self.max_iterations),
                    'stages': int(max_stages)
                },
                'convergence_history': {
                    'costs': [float(c) for c in self.cost_history],
                    'temperatures': [float(t) for t in self.temperature_history],
                    'acceptance_rates': [float(r) for r in self.acceptance_rates]
                }
            }

            self._save_report(results)
            self.create_visualizations()

            return results

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            execution_time = time.time() - self.start_time
            return {
                'performance_metrics': {
                    'makespan': float('inf'),
                    'best_cost': float('inf'),
                    'execution_time': float(execution_time),
                    'iterations': 0,
                    'violations': {'precedence': 0, 'resource': 0}
                },
                'schedule': [],
                'error': str(e)
            }

    def _apply_perturbation(self, schedule: List[int]) -> List[int]:
        """Apply strong perturbation to escape local optima"""
        for _ in range(20):  # Try multiple times
            try:
                perturbed = schedule.copy()

                # Choose strong perturbation type
                perturbation = random.choice([
                    'multiple_swaps',
                    'large_block_move',
                    'multiple_reversals',
                    'random_reconstruction'
                ])

                if perturbation == 'multiple_swaps':
                    # Multiple random swaps
                    num_swaps = random.randint(5, 10)
                    for _ in range(num_swaps):
                        i, j = random.sample(range(len(perturbed)), 2)
                        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

                elif perturbation == 'large_block_move':
                    # Move large block of tasks
                    block_size = len(perturbed) // 4
                    start = random.randint(0, len(perturbed) - block_size)
                    block = perturbed[start:start + block_size]
                    del perturbed[start:start + block_size]
                    insert_point = random.randint(0, len(perturbed))
                    perturbed[insert_point:insert_point] = block

                elif perturbation == 'multiple_reversals':
                    # Multiple segment reversals
                    for _ in range(3):
                        i, j = sorted(random.sample(range(len(perturbed)), 2))
                        if j - i > 2:
                            perturbed[i:j + 1] = reversed(perturbed[i:j + 1])

                else:  # random_reconstruction
                    # Reconstruct part of the solution randomly
                    preserve_size = len(perturbed) // 2
                    preserved = perturbed[:preserve_size]
                    remaining = list(set(range(self.num_tasks)) - set(preserved))
                    random.shuffle(remaining)
                    perturbed = preserved + remaining

                if self._is_valid_schedule(perturbed):
                    return perturbed

            except Exception as e:
                print(f"Error in perturbation: {str(e)}")
                continue

        return schedule  # Return original if no valid perturbation found

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



def main():
    try:
        # Choose dataset size (30, 60, 90, or 120)
        dataset_size = "60"
        json_dir = os.path.join('processed_data', f'j{dataset_size}.sm', 'json')
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        if not json_files:
            raise ValueError(f"No JSON files found in {json_dir}")

        dataset_path = os.path.join(json_dir, json_files[0])
        print(f"Using dataset: {dataset_path}")

        scheduler = SimulatedAnnealingScheduler(dataset_path)
        results = scheduler.optimize()

        print("\nResults:")
        print(f"Makespan: {results['performance_metrics']['makespan']:.2f}")
        print(f"Best Cost: {results['performance_metrics']['best_cost']:.2f}")
        print(f"Execution Time: {results['performance_metrics']['execution_time']:.2f} seconds")
        print(f"Precedence Violations: {results['performance_metrics']['violations']['precedence']}")
        print(f"Resource Violations: {results['performance_metrics']['violations']['resource']}")
        print(f"\nResults and visualizations saved in: {scheduler.output_dir}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()