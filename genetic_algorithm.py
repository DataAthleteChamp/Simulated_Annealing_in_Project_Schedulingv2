import numpy as np
import matplotlib.pyplot as plt
import json
import random
import time
from typing import List, Dict, Tuple
import os
from datetime import datetime
import traceback
from multiprocessing import Pool


class GeneticScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the Genetic Algorithm scheduler with dataset"""
        self._load_dataset(dataset_path)
        self._initialize_tracking()
        self._create_output_directories()
        self.params = self._tune_parameters()

    def _tune_parameters(self) -> Dict:
        """Tune GA parameters for better exploration and consistency"""
        if self.num_tasks < 31:  # j30
            population_size = 50
            generations = 150
        elif self.num_tasks < 61:  # j60
            population_size = 100
            generations = 300  # Increased from 200
        else:  # j90/j120
            population_size = 150
            generations = 400

        return {
            'population_size': population_size,
            'num_generations': generations,
            'mutation_rate': 0.1,  # Increased from 0.05
            'crossover_rate': 0.85,
            'elite_size': max(2, population_size // 10),
            'tournament_size': 4,
            'stagnation_limit': 50,  # Increased from 30
            'diversity_threshold': 0.3
        }

    def _calculate_fitness(self, schedule: List[int]) -> float:
        """Enhanced fitness calculation focusing on makespan"""
        if not self._is_valid_schedule(schedule):
            return 0.0

        task_times = self._calculate_task_times(schedule)
        makespan = max(timing['end'] for timing in task_times.values())

        # Prioritize makespan reduction more strongly
        return 1.0 / makespan

    def _get_default_parameters(self) -> Dict:
        """Provide safe default parameters"""
        return {
            'population_size': 100,
            'num_generations': 100,
            'mutation_rate': 0.05,
            'crossover_rate': 0.8,
            'elite_size': 5,
            'tournament_size': 5,
            'diversity_threshold': 0.3,
            'stagnation_limit': 20
        }

    def _initialize_population(self) -> List[List[int]]:
        """Initialize population with valid schedules"""
        population = []
        for _ in range(self.params['population_size']):
            schedule = self._create_valid_schedule()
            population.append(schedule)
        return population

    def _create_valid_schedule(self) -> List[int]:
        """Create a valid schedule with improved randomization"""
        in_degree = {i: 0 for i in range(self.num_tasks)}
        adj_list = {i: [] for i in range(self.num_tasks)}

        # Build adjacency list and calculate in-degrees
        for i, task in enumerate(self.tasks):
            for succ in task.get('successors', []):
                adj_list[i].append(succ)
                in_degree[succ] += 1

        available = [i for i, deg in in_degree.items() if deg == 0]
        schedule = []

        while available:
            # Calculate priority scores for available tasks
            weights = []
            for task in available:
                # Normalize and combine multiple factors
                processing_time = self.tasks[task]['processing_time']
                resource_demand = sum(self.tasks[task]['resource_requirements'].values())
                successor_count = len(self.tasks[task].get('successors', []))

                # Ensure positive weights by adding a small constant
                weight = 1.0 + (
                        0.3 * processing_time +
                        0.3 * resource_demand +
                        0.4 * successor_count
                )
                weights.append(weight)

            # Select task based on weights
            selected_idx = random.choices(range(len(available)), weights=weights)[0]
            task = available.pop(selected_idx)
            schedule.append(task)

            # Update successors
            for succ in adj_list[task]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    available.append(succ)

        return schedule

    def _calculate_temporal_distribution(self, task_times: Dict) -> float:
        """Calculate how well tasks are distributed across the schedule timeline"""
        if not task_times:
            return 0.0

        makespan = max(timing['end'] for timing in task_times.values())
        if makespan == 0:
            return 0.0

        # Create timeline of task density
        timeline = [0] * (int(makespan) + 1)
        for timing in task_times.values():
            start = int(timing['start'])
            end = int(timing['end'])
            for t in range(start, end):
                timeline[t] += 1

        # Calculate variance in task distribution
        avg_tasks = sum(timeline) / len(timeline)
        variance = sum((tasks - avg_tasks) ** 2 for tasks in timeline) / len(timeline)

        # Lower variance means better distribution
        distribution_score = 1.0 / (1.0 + variance)

        return distribution_score

    # def _calculate_fitness(self, schedule: List[int]) -> float:
    #     """Calculate fitness with improved scaling and distribution metrics"""
    #     if not self._is_valid_schedule(schedule):
    #         return 0.0
    #
    #     task_times = self._calculate_task_times(schedule)
    #     makespan = max(timing['end'] for timing in task_times.values())
    #
    #     # Calculate makespan component
    #     makespan_score = (1.0 / makespan) ** 2
    #
    #     # Calculate resource utilization
    #     resource_utilization = self._calculate_resource_utilization_ratio(schedule, task_times)
    #
    #     # Calculate temporal distribution
    #     distribution_score = self._calculate_temporal_distribution(task_times)
    #
    #     # Combine scores with weights
    #     fitness = (0.6 * makespan_score +
    #                0.2 * resource_utilization +
    #                0.2 * distribution_score)
    #
    #     return fitness

    def _calculate_resource_utilization_ratio(self, schedule: List[int],
                                              task_times: Dict) -> float:
        """Calculate average resource utilization across all resources"""
        if not schedule or not task_times:
            return 0.0

        makespan = max(timing['end'] for timing in task_times.values())
        total_utilization = 0.0

        for resource in self.global_resources:
            capacity = self.global_resources[resource]
            total_available = capacity * makespan
            total_used = 0

            for task_id in schedule:
                task = self.tasks[task_id]
                duration = task_times[task_id]['end'] - task_times[task_id]['start']
                resource_demand = task['resource_requirements'][resource]
                total_used += duration * resource_demand

            resource_utilization = total_used / total_available if total_available > 0 else 0
            total_utilization += resource_utilization

        return total_utilization / len(self.global_resources)

    def _tournament_selection(self, population: List[List[int]],
                              fitness_scores: List[float]) -> List[int]:
        """Select parent using tournament selection"""
        tournament_size = self.params['tournament_size']
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform precedence-preserving crossover"""
        if random.random() > self.params['crossover_rate']:
            return parent1.copy()

        # Create task position mappings
        pos1 = {task: idx for idx, task in enumerate(parent1)}
        pos2 = {task: idx for idx, task in enumerate(parent2)}

        # Choose crossover point
        cross_point = random.randint(1, len(parent1) - 2)

        # Initialize offspring
        offspring = [-1] * len(parent1)
        used_tasks = set()

        # Copy first part from parent1
        for i in range(cross_point):
            offspring[i] = parent1[i]
            used_tasks.add(parent1[i])

        # Fill remaining positions while preserving precedence
        remaining_tasks = [t for t in parent2 if t not in used_tasks]
        current_pos = cross_point

        for task in remaining_tasks:
            # Check if task can be placed at current position
            while current_pos < len(offspring):
                if self._can_place_task(task, current_pos, offspring):
                    offspring[current_pos] = task
                    current_pos += 1
                    break
                current_pos += 1

        return offspring

    def _can_place_task(self, task: int, position: int,
                        partial_schedule: List[int]) -> bool:
        """Check if task can be placed at position while preserving precedence"""
        # Check predecessors
        for i, placed_task in enumerate(partial_schedule[:position]):
            if placed_task == -1:
                continue
            if task in self.tasks[placed_task].get('successors', []):
                return False

        # Check successors
        for i, placed_task in enumerate(partial_schedule[:position]):
            if placed_task == -1:
                continue
            if placed_task in self.tasks[task].get('successors', []):
                return True

        return True

    def _mutate(self, schedule: List[int], strong_mutation: bool = False) -> List[int]:
        """Perform precedence-preserving mutation with adjustable strength"""
        # Check if mutation should be applied
        if not strong_mutation and random.random() > self.params['mutation_rate']:
            return schedule

        mutated = schedule.copy()
        iterations = 5 if strong_mutation else 1
        max_attempts = 20 if strong_mutation else 10

        for _ in range(iterations):
            attempts = 0
            while attempts < max_attempts:
                # Choose two random positions
                i, j = random.sample(range(len(schedule)), 2)

                # Check if swap preserves precedence constraints
                if self._is_valid_swap(mutated, i, j):
                    mutated[i], mutated[j] = mutated[j], mutated[i]
                    if self._is_valid_schedule(mutated):
                        break

                attempts += 1

                # If strong mutation, try additional mutation types
                if strong_mutation and attempts == max_attempts // 2:
                    # Try block reversal
                    start = random.randint(0, len(mutated) - 3)
                    end = random.randint(start + 2, min(start + 5, len(mutated)))
                    segment = mutated[start:end]
                    segment.reverse()
                    test_schedule = mutated[:start] + segment + mutated[end:]
                    if self._is_valid_schedule(test_schedule):
                        mutated = test_schedule
                        break

        return mutated

    def _is_valid_swap(self, schedule: List[int], i: int, j: int) -> bool:
        """Check if swapping tasks at positions i and j preserves precedence"""
        task1, task2 = schedule[i], schedule[j]

        # Check if task2 is successor of task1
        if task2 in self.tasks[task1].get('successors', []):
            return False

        # Check if task1 is successor of task2
        if task1 in self.tasks[task2].get('successors', []):
            return False

        return True

    def _fast_validation(self, schedule: List[int]) -> bool:
        """Validate schedule focusing on precedence constraints"""
        # First verify all tasks are present exactly once
        if len(schedule) != self.num_tasks or len(set(schedule)) != self.num_tasks:
            return False

        task_positions = {task_id: pos for pos, task_id in enumerate(schedule)}

        # Verify precedence constraints
        for task_id in schedule:
            task = self.tasks[task_id]
            for succ in task.get('successors', []):
                if succ not in task_positions:
                    return False
                if task_positions[succ] < task_positions[task_id]:
                    return False

        return True

    def _fast_mutate(self, schedule: List[int]) -> List[int]:
        """Efficient mutation operator that maintains precedence constraints"""
        for _ in range(3):  # Try a few times to find a valid mutation
            mutated = schedule.copy()
            i, j = random.sample(range(len(schedule)), 2)

            # Check if swap would violate precedence
            task1, task2 = mutated[i], mutated[j]

            # Simple precedence check before swap
            if (task2 not in self.tasks[task1].get('successors', []) and
                    task1 not in self.tasks[task2].get('successors', [])):
                mutated[i], mutated[j] = mutated[j], mutated[i]
                if self._fast_validation(mutated):
                    return mutated

        return schedule

    def optimize(self) -> Dict:
        """Execute genetic algorithm optimization with improved efficiency"""
        print("\nStarting genetic algorithm optimization...")
        self.start_time = time.time()

        try:
            # Initialize population
            population = self._initialize_population()
            best_fitness = 0.0
            best_schedule = None
            generations_without_improvement = 0

            # Main evolutionary loop
            for generation in range(self.params['num_generations']):
                fitness_scores = [self._calculate_fitness(schedule) for schedule in population]

                # Update best solution
                max_fitness = max(fitness_scores)
                if max_fitness > best_fitness:
                    best_fitness = max_fitness
                    best_schedule = population[fitness_scores.index(max_fitness)]
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1

                # Track progress
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                self.generation_history['best_fitness'].append(max_fitness)
                self.generation_history['avg_fitness'].append(avg_fitness)

                # Check for early stopping
                if generations_without_improvement >= self.params['stagnation_limit']:
                    print(f"\nStopping early at generation {generation} due to stagnation")
                    break

                # Create new population
                new_population = []

                # Preserve elite solutions
                elite_indices = sorted(range(len(fitness_scores)),
                                       key=lambda i: fitness_scores[i],
                                       reverse=True)[:self.params['elite_size']]
                new_population.extend([population[i].copy() for i in elite_indices])

                # Generate offspring
                while len(new_population) < self.params['population_size']:
                    # Select parents using tournament selection
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)

                    # Create and potentially mutate offspring
                    offspring = self._crossover(parent1, parent2)
                    if random.random() < self.params['mutation_rate']:
                        offspring = self._fast_mutate(offspring)

                    # Add valid offspring to new population
                    if self._fast_validation(offspring):
                        new_population.append(offspring)

                population = new_population

                # Inside the optimize method, replace the progress reporting section with:
                if generation % 10 == 0:
                    actual_makespan = \
                    max(self._calculate_task_times(population[fitness_scores.index(max_fitness)]).values(),
                        key=lambda x: x['end'])['end']
                    print(f"Generation {generation}:")
                    print(f"  Actual Makespan = {actual_makespan:.2f}")
                    print(f"  Best Fitness = {max_fitness:.4f}")

                # # Report progress
                # if generation % 10 == 0:
                #     makespan = 1.0 / max_fitness if max_fitness > 0 else float('inf')
                #     print(f"Generation {generation}: Best Makespan = {makespan:.2f}")

            # Prepare final results
            final_schedule = self._calculate_final_schedule(best_schedule)
            results = self._prepare_results(final_schedule)

            self._save_report(results)
            self.create_visualizations(results)

            return results

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            traceback.print_exc()
            return self._prepare_error_results()

    def _inject_diversity(self, population: List[List[int]], fitness_scores: List[float]) -> List[List[int]]:
        """Inject diversity into the population when stagnation is detected"""
        new_population = []

        # Keep top 20% of existing solutions
        elite_size = len(population) // 5
        elite_indices = sorted(range(len(fitness_scores)),
                               key=lambda i: fitness_scores[i],
                               reverse=True)[:elite_size]
        new_population.extend([population[i].copy() for i in elite_indices])

        # Generate new random solutions for the rest
        while len(new_population) < len(population):
            schedule = self._create_valid_schedule()
            new_population.append(schedule)

        return new_population

    def _calculate_final_schedule(self, schedule: List[int]) -> List[Dict]:
        """Convert chromosome to detailed schedule"""
        if not schedule:
            return []

        task_times = self._calculate_task_times(schedule)
        final_schedule = []

        for task_id in schedule:
            timing = task_times[task_id]
            final_schedule.append({
                'task_id': task_id,
                'start_time': float(timing['start']),
                'end_time': float(timing['end']),
                'processing_time': float(self.tasks[task_id]['processing_time'])
            })

        return sorted(final_schedule, key=lambda x: x['start_time'])

    def create_visualizations(self, results: Dict):
        """Generate visualizations"""
        try:
            self._plot_schedule(results['schedule'])
            self._plot_resource_utilization(results['schedule'])
            if self.generation_history['best_fitness']:  # Only plot if we have data
                self._plot_convergence(results['convergence_history'])
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

    def _plot_convergence(self, history: Dict):
        """Plot fitness convergence over generations"""
        plt.figure(figsize=(12, 6))
        plt.plot(history['best_fitness'], 'b-', label='Best Fitness')
        plt.plot(history['avg_fitness'], 'r--', label='Average Fitness')
        plt.title('Fitness Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.viz_dir, 'convergence.png'))
        plt.close()

    def _plot_schedule(self, schedule: List[Dict]):
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

        plt.title('Project Schedule (Gantt Chart)')
        plt.xlabel('Time')
        plt.ylabel('Task ID')
        plt.grid(True, alpha=0.3)

        legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6)
                          for color in resource_colors.values()]
        plt.legend(legend_patches, list(self.global_resources.keys()),
                   title='Main Resource', loc='center left',
                   bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'schedule.png'),
                    bbox_inches='tight')
        plt.close()

    def _load_dataset(self, dataset_path: str):
        """Load and validate the dataset from JSON file"""
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
        """Initialize tracking variables for genetic algorithm"""
        self.best_schedule = None
        self.best_cost = float('inf')
        self.start_time = None
        self.current_violations = {'precedence': 0, 'resource': 0}

        # Initialize generation history with all required keys
        self.generation_history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],  # Added diversity tracking
            'population_diversity': []
        }

        self.current_generation = 0
        self.best_solutions = []

    def _create_output_directories(self):
        """Create directories for output files and visualizations"""
        try:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"genetic_results_{self.timestamp}"
            self.viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.viz_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error creating directories: {str(e)}")

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

    def _is_valid_schedule(self, schedule: List[int]) -> bool:
        """Validate schedule feasibility considering precedence and resource constraints"""
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
            task_times = self._calculate_task_times(schedule)
            resource_usage = {}

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

    def _calculate_makespan(self, schedule: List[Dict]) -> float:
        """Calculate makespan of final schedule"""
        try:
            if not schedule:
                return float('inf')
            return max(task['end_time'] for task in schedule)
        except Exception as e:
            print(f"Error calculating makespan: {str(e)}")
            return float('inf')

    def _prepare_results(self, final_schedule: List[Dict]) -> Dict:
        execution_time = time.time() - self.start_time
        makespan = max(task['end_time'] for task in final_schedule) if final_schedule else float('inf')

        best_fitness = max(self.generation_history['best_fitness']) if self.generation_history['best_fitness'] else 0.0

        return {
            'performance_metrics': {
                'makespan': float(makespan),
                'execution_time': float(execution_time),
                'generations': len(self.generation_history['best_fitness']),
                'population_size': self.params['population_size'],
                'best_fitness': float(best_fitness),
                'violations': self.current_violations
            },
            'schedule': final_schedule,
            'algorithm_parameters': self.params,
            'convergence_history': {
                'best_fitness': [float(f) for f in self.generation_history['best_fitness']],
                'avg_fitness': [float(f) for f in self.generation_history['avg_fitness']]
            }
        }

    def _prepare_error_results(self) -> Dict:
        """Prepare error results structure"""
        execution_time = time.time() - self.start_time if self.start_time else 0

        return {
            'performance_metrics': {
                'makespan': float('inf'),
                'execution_time': float(execution_time),
                'generations': 0,
                'population_size': self.params['population_size'],
                'best_fitness': 0.0,
                'violations': {'precedence': 0, 'resource': 0}
            },
            'schedule': [],
            'algorithm_parameters': self.params,
            'convergence_history': {
                'best_fitness': [],
                'avg_fitness': [],
                'diversity': []
            },
            'error': 'Optimization failed'
        }

    def _plot_resource_utilization(self, schedule: List[Dict]):
        """Plot resource utilization over time"""
        makespan = max(task['end_time'] for task in schedule)
        timeline = {t: {r: 0 for r in self.global_resources}
                    for t in range(int(makespan) + 1)}

        # Calculate resource usage
        for task in schedule:
            start = int(task['start_time'])
            end = int(task['end_time'])
            task_id = task['task_id']

            for t in range(start, end):
                for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                    timeline[t][resource] += amount

        # Create visualization
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

    def _save_report(self, result: Dict):
        """Save the analysis report"""
        try:
            report_path = os.path.join(self.output_dir, 'genetic_report.json')

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

            return max(early_finish)

        except Exception as e:
            print(f"Error calculating critical path: {str(e)}")
            return self.num_tasks


def main():
    try:
        dataset_size = "60"
        json_dir = os.path.join('processed_data', f'j{dataset_size}.sm', 'json')
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        if not json_files:
            raise ValueError(f"No JSON files found in {json_dir}")

        dataset_path = os.path.join(json_dir, json_files[0])
        print(f"Using dataset: {dataset_path}")

        scheduler = GeneticScheduler(dataset_path)
        results = scheduler.optimize()

        print("\nOptimization Results:")
        print(f"Makespan: {results['performance_metrics']['makespan']:.2f}")
        print(f"Execution Time: {results['performance_metrics']['execution_time']:.2f} seconds")
        print(f"Generations: {results['performance_metrics']['generations']}")
        print(f"Final Population Size: {results['performance_metrics']['population_size']}")
        print(f"Best Fitness: {results['performance_metrics']['best_fitness']:.4f}")
        print(f"\nResults and visualizations saved in: {scheduler.output_dir}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()