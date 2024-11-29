import numpy as np
import random
import time
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import traceback
from dataclasses import dataclass


@dataclass
class GeneticParams:
    population_size: int
    mutation_rate: float
    crossover_rate: float
    tournament_size: int
    elite_size: int
    max_generations: int
    convergence_threshold: float


class GeneticScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the Genetic Algorithm scheduler with dataset"""
        self._load_dataset(dataset_path)
        self._initialize_tracking()
        self._create_output_directories()
        self.params = self._tune_parameters()

    def _tune_parameters(self) -> GeneticParams:
        """Tune GA parameters adaptively based on problem size and characteristics"""
        try:
            # Calculate problem characteristics
            resource_complexity = self._calculate_resource_complexity()
            network_density = self._calculate_network_density()
            critical_path_length = self._calculate_critical_path_length()

            # Base parameters based on problem size
            if self.num_tasks < 34:  # j30
                base_params = {
                    'population_size': 100,
                    'generations': 200,
                    'tournament_size': 3,
                    'elite_ratio': 0.05,
                    'mutation_base': 0.04
                }
            elif self.num_tasks < 64:  # j60
                base_params = {
                    'population_size': 200,
                    'generations': 300,
                    'tournament_size': 4,
                    'elite_ratio': 0.07,
                    'mutation_base': 0.05
                }
            elif self.num_tasks < 94:  # j90
                base_params = {
                    'population_size': 300,
                    'generations': 400,
                    'tournament_size': 5,
                    'elite_ratio': 0.08,
                    'mutation_base': 0.06
                }
            else:  # j120
                base_params = {
                    'population_size': 400,
                    'generations': 500,
                    'tournament_size': 6,
                    'elite_ratio': 0.1,
                    'mutation_base': 0.07
                }

            # Adjust population size based on complexity
            complexity_factor = 1 + (resource_complexity * 0.5) + ((1 - network_density) * 0.3)
            population_size = int(base_params['population_size'] * complexity_factor)

            # Adjust mutation rate based on problem characteristics
            mutation_rate = max(0.01, min(0.15,
                                          base_params['mutation_base'] * (1 + (1 - network_density) * 0.5)))

            # Adjust crossover rate based on problem size
            crossover_rate = min(0.95, 0.85 + self.num_tasks / 1000)

            # Calculate tournament size based on population
            tournament_size = max(
                base_params['tournament_size'],
                min(8, population_size // 40)
            )

            # Calculate elite size based on population
            elite_size = max(
                2,
                int(population_size * base_params['elite_ratio'])
            )

            # Adjust generations based on problem size and complexity
            max_generations = int(base_params['generations'] *
                                  (1 + (resource_complexity * 0.2)))

            # Early stopping parameters
            no_improve_limit = max(20, self.num_tasks // 3)

            # Print tuned parameters for reference
            print("\nTuned Parameters:")
            print(f"Population Size: {population_size}")
            print(f"Mutation Rate: {mutation_rate:.4f}")
            print(f"Crossover Rate: {crossover_rate:.4f}")
            print(f"Tournament Size: {tournament_size}")
            print(f"Elite Size: {elite_size}")
            print(f"Max Generations: {max_generations}")
            print(f"Early Stopping Limit: {no_improve_limit}")

            return GeneticParams(
                population_size=population_size,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                tournament_size=tournament_size,
                elite_size=elite_size,
                max_generations=max_generations,
                convergence_threshold=0.001
            )

        except Exception as e:
            print(f"Error in parameter tuning: {str(e)}")
            return self._get_default_parameters()

    def _get_default_parameters(self) -> GeneticParams:
        """Provide safe default parameters if tuning fails"""
        return GeneticParams(
            population_size=200,
            mutation_rate=0.05,
            crossover_rate=0.90,
            tournament_size=4,
            elite_size=10,
            max_generations=300,
            convergence_threshold=0.001
        )

    def _create_initial_population(self) -> List[List[int]]:
        """Create initial population using various heuristics"""
        population = []

        for _ in range(self.params.population_size):
            if random.random() < 0.7:
                # Create solution using topological sort with randomization
                solution = self._create_topological_solution()
            else:
                # Create solution using resource-based priority
                solution = self._create_resource_based_solution()

            if self._is_valid_schedule(solution):
                population.append(solution)

        # Fill remaining slots with random valid solutions
        while len(population) < self.params.population_size:
            solution = self._create_random_solution()
            if self._is_valid_schedule(solution):
                population.append(solution)

        return population

    def _create_topological_solution(self) -> List[int]:
        """Create solution using randomized topological sort"""
        in_degree = {i: 0 for i in range(self.num_tasks)}
        for task_id, task in enumerate(self.tasks):
            for succ in task.get('successors', []):
                in_degree[succ] += 1

        available = [i for i, degree in in_degree.items() if degree == 0]
        schedule = []

        while available:
            # Randomly select from available tasks with weights
            weights = []
            for task in available:
                weight = 1.0
                # Prefer shorter processing times
                weight *= 1.0 / (self.tasks[task]['processing_time'] + 1)
                # Prefer less resource usage
                weight *= 1.0 / (sum(self.tasks[task]['resource_requirements'].values()) + 1)
                weights.append(weight)

            task = random.choices(available, weights=weights)[0]
            schedule.append(task)
            available.remove(task)

            # Update available tasks
            for succ in self.tasks[task].get('successors', []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    available.append(succ)

        return schedule

    def _create_resource_based_solution(self) -> List[int]:
        """Create solution prioritizing resource balance"""
        available = set(range(self.num_tasks))
        schedule = []
        resource_usage = {r: 0 for r in self.global_resources}

        while available:
            candidates = []
            for task in available:
                if all(succ not in available for succ in self.tasks[task].get('successors', [])):
                    candidates.append(task)

            if not candidates:
                break

            # Score candidates based on resource balance
            scores = []
            for task in candidates:
                score = 0
                for resource, amount in self.tasks[task]['resource_requirements'].items():
                    balance = resource_usage[resource] / self.global_resources[resource]
                    score -= abs(0.5 - balance)  # Prefer balanced usage
                scores.append(score)

            task = random.choices(candidates, weights=[s - min(scores) + 1 for s in scores])[0]
            schedule.append(task)
            available.remove(task)

            # Update resource usage
            for resource, amount in self.tasks[task]['resource_requirements'].items():
                resource_usage[resource] += amount

        return schedule

    def _tournament_selection(self, population: List[List[int]], fitness: List[float]) -> List[int]:
        """Select parent using tournament selection"""
        tournament = random.sample(range(len(population)), self.params.tournament_size)
        winner = min(tournament, key=lambda i: fitness[i])
        return population[winner]

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform order crossover (OX) while preserving precedence constraints"""
        if random.random() > self.params.crossover_rate:
            return parent1

        size = len(parent1)
        # Select crossover points
        start, end = sorted(random.sample(range(size), 2))

        # Create child using segment from parent1
        child = [-1] * size
        segment = parent1[start:end]
        child[start:end] = segment

        # Fill remaining positions using order from parent2
        pos = end
        for item in parent2:
            if item not in segment:
                if pos >= size:
                    pos = 0
                while pos >= start and pos < end:
                    pos += 1
                    if pos >= size:
                        pos = 0
                child[pos] = item
                pos += 1

        return child

    def _adaptive_mutation(self, schedule: List[int], generation: int) -> List[int]:
        """Apply adaptive mutation based on search progress"""
        if random.random() > self.params.mutation_rate:
            return schedule

        mutated = schedule.copy()
        num_mutations = random.randint(1, 3)

        for _ in range(num_mutations):
            mutation_type = random.choices(
                ['swap', 'insert', 'reverse'],
                weights=[0.5, 0.3, 0.2]
            )[0]

            if mutation_type == 'swap':
                i, j = random.sample(range(len(mutated)), 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]

            elif mutation_type == 'insert':
                i = random.randint(0, len(mutated) - 1)
                j = random.randint(0, len(mutated) - 1)
                task = mutated.pop(i)
                mutated.insert(j, task)

            else:  # reverse
                i, j = sorted(random.sample(range(len(mutated)), 2))
                mutated[i:j] = reversed(mutated[i:j])

        return mutated

    def _analyze_resource_utilization(self, schedule: List[Dict]) -> Dict:
        """Analyze resource utilization percentages and patterns - report only"""
        try:
            makespan = max(task['end_time'] for task in schedule)
            timeline = {t: {r: 0 for r in self.global_resources}
                        for t in range(int(makespan) + 1)}

            # Calculate resource usage over time
            for task in schedule:
                task_id = task['task_id']
                start = int(task['start_time'])
                end = int(task['end_time'])

                for t in range(start, end):
                    for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                        timeline[t][resource] += amount

            # Calculate utilization statistics
            utilization_stats = {}
            for resource, capacity in self.global_resources.items():
                usage_values = [timeline[t][resource] for t in range(int(makespan) + 1)]
                avg_usage = sum(usage_values) / len(usage_values)
                max_usage = max(usage_values)
                utilization_percent = (avg_usage / capacity) * 100
                peak_utilization = (max_usage / capacity) * 100

                utilization_stats[resource] = {
                    'average_usage': avg_usage,
                    'max_usage': max_usage,
                    'utilization_percent': utilization_percent,
                    'peak_utilization': peak_utilization,
                    'capacity': capacity
                }

            return utilization_stats

        except Exception as e:
            print(f"Error analyzing resource utilization: {str(e)}")
            return {}

    def optimize(self) -> Dict:
        """Main genetic algorithm optimization loop"""
        print("\nStarting genetic algorithm optimization...")
        self.start_time = time.time()

        try:
            # Initialize population
            population = self._create_initial_population()
            best_solution = None
            best_fitness = float('inf')
            generations_without_improvement = 0

            for generation in range(self.params.max_generations):
                # Evaluate population
                fitness_scores = [self._calculate_cost(schedule)[0] for schedule in population]

                # Update best solution
                min_fitness = min(fitness_scores)
                if min_fitness < best_fitness:
                    best_fitness = min_fitness
                    best_solution = population[fitness_scores.index(min_fitness)]
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1

                # Early stopping
                if generations_without_improvement > 20:
                    print(f"\nStopping early at generation {generation} due to no improvement")
                    break

                # Create new population
                new_population = []

                # Elitism
                elite = sorted(zip(fitness_scores, population), key=lambda x: x[0])
                new_population.extend([ind for _, ind in elite[:self.params.elite_size]])

                # Generate offspring
                while len(new_population) < self.params.population_size:
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)

                    # Crossover
                    offspring = self._order_crossover(parent1, parent2)

                    # Mutation
                    offspring = self._adaptive_mutation(offspring, generation)

                    if self._is_valid_schedule(offspring):
                        new_population.append(offspring)

                # Update tracking metrics
                self._update_tracking(min_fitness, generation, new_population)

                # Progress report
                if generation % 10 == 0:
                    print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")

                population = new_population

            # Prepare final results
            self.best_solution = best_solution
            final_schedule = self._calculate_final_schedule(best_solution)
            results = self._prepare_results(final_schedule)

            # Save results and visualizations
            self._save_report(results)
            self.create_visualizations()

            return results

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            traceback.print_exc()
            return self._prepare_error_results()

    def create_visualizations(self):
        """Generate all visualizations with error handling"""
        try:
            # Create visualizations directory
            os.makedirs(self.viz_dir, exist_ok=True)

            # Convergence plot
            plt.figure(figsize=(12, 6))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.plot(self.generation_history, self.cost_history, 'b-', linewidth=2, alpha=0.7, label='Best Fitness')
            plt.plot([gen for _, gen in self.best_solutions],
                     [fit for fit, _ in self.best_solutions],
                     'ro', label='New Best Solution', markersize=8)

            plt.title('Genetic Algorithm Convergence', pad=20, fontsize=12)
            plt.xlabel('Generation', fontsize=10)
            plt.ylabel('Fitness (Makespan)', fontsize=10)
            plt.legend(frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'convergence.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Population diversity plot
            if self.diversity_history:
                plt.figure(figsize=(12, 6))
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.plot(self.generation_history, self.diversity_history, 'g-',
                         linewidth=2, alpha=0.7, label='Population Diversity')
                plt.title('Population Diversity Over Time', pad=20, fontsize=12)
                plt.xlabel('Generation', fontsize=10)
                plt.ylabel('Diversity Metric', fontsize=10)
                plt.legend(frameon=True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.viz_dir, 'diversity.png'), dpi=300, bbox_inches='tight')
                plt.close()

            # Gantt chart
            if self.best_solution:
                final_schedule = self._calculate_final_schedule(self.best_solution)
                self._plot_best_schedule(final_schedule)
                self._plot_resource_utilization(final_schedule)

            print(f"Visualizations saved in: {self.viz_dir}")

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            traceback.print_exc()

    def _update_tracking(self, current_fitness: float, generation: int, population: List[List[int]]):
        """Update optimization tracking metrics"""
        self.cost_history.append(float(current_fitness))
        self.generation_history.append(generation)

        # Calculate and store population diversity
        diversity = self._calculate_population_diversity(population)
        self.diversity_history.append(diversity)

        if len(self.best_solutions) == 0 or current_fitness < self.best_solutions[-1][0]:
            self.best_solutions.append((current_fitness, generation))
            self.best_solution = population[self.cost_history.index(current_fitness)]

    def _plot_convergence(self):
        """Plot optimization convergence history"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.generation_history, self.cost_history, 'b-', alpha=0.5, label='Best Fitness')
        plt.plot([gen for _, gen in self.best_solutions],
                 [fit for fit, _ in self.best_solutions],
                 'r.', label='New Best Solution')

        plt.title('Genetic Algorithm Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Makespan)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.savefig(os.path.join(self.viz_dir, 'convergence.png'))
        plt.close()

    def _plot_population_diversity(self):
        """Plot population diversity over generations"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.diversity_history, 'g-', label='Population Diversity')

        plt.title('Population Diversity Over Time')
        plt.xlabel('Generation')
        plt.ylabel('Diversity Metric')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.savefig(os.path.join(self.viz_dir, 'diversity.png'))
        plt.close()

    def _plot_best_schedule(self, final_schedule: List[Dict]):
        """Create Gantt chart with viridis color scheme"""
        if not final_schedule:
            return

        plt.figure(figsize=(15, 8))
        plt.grid(True, linestyle='--', alpha=0.3)

        # Use viridis colormap as before
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.global_resources)))
        resource_colors = dict(zip(self.global_resources.keys(), colors))

        # Calculate y-axis limits
        tasks = list(range(self.num_tasks))
        plt.ylim(-1, self.num_tasks)

        # Calculate x-axis limits
        max_time = max(task['end_time'] for task in final_schedule)
        plt.xlim(0, max_time * 1.05)

        for task in final_schedule:
            task_id = task['task_id']
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

        plt.title('Best Schedule (Gantt Chart)')
        plt.xlabel('Time')
        plt.ylabel('Task ID')

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'best_schedule.png'),
                    bbox_inches='tight')
        plt.close()

    def _plot_resource_utilization(self, final_schedule: List[Dict]):
        """Create resource utilization plot with original style"""
        if not final_schedule:
            return

        makespan = max(task['end_time'] for task in final_schedule)
        timeline = {t: {r: 0 for r in self.global_resources}
                    for t in range(int(makespan) + 1)}

        # Calculate resource usage
        for task in final_schedule:
            task_id = task['task_id']
            start = int(task['start_time'])
            end = int(task['end_time'])

            for t in range(start, end):
                for resource, amount in self.tasks[task_id]['resource_requirements'].items():
                    timeline[t][resource] += amount

        # Create plot
        plt.figure(figsize=(15, 8))
        plt.grid(True, alpha=0.3)

        times = list(range(int(makespan) + 1))

        for resource in self.global_resources:
            usage = [timeline[t][resource] for t in times]
            plt.plot(times, usage, label=f'{resource} Usage', alpha=0.7)
            plt.axhline(y=self.global_resources[resource],
                        color='red', linestyle='--', alpha=0.3,
                        label=f'{resource} Capacity')

        plt.title('Resource Utilization Profile')
        plt.xlabel('Time')
        plt.ylabel('Resource Units')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'resource_utilization.png'),
                    bbox_inches='tight')
        plt.close()

    def _prepare_results(self, final_schedule: List[Dict]) -> Dict:
        """Prepare comprehensive results dictionary including resource analysis"""
        execution_time = time.time() - self.start_time
        makespan = self._calculate_makespan(final_schedule)

        # Calculate resource utilization stats
        utilization_stats = self._analyze_resource_utilization(final_schedule)

        return {
            'performance_metrics': {
                'makespan': float(makespan),
                'best_fitness': float(min(self.cost_history)),
                'execution_time': float(execution_time),
                'generations': len(self.generation_history),
                'population_size': self.params.population_size,
                'improvements': len(self.best_solutions)
            },
            'schedule': final_schedule,
            'algorithm_parameters': {
                'population_size': self.params.population_size,
                'mutation_rate': self.params.mutation_rate,
                'crossover_rate': self.params.crossover_rate,
                'tournament_size': self.params.tournament_size,
                'elite_size': self.params.elite_size,
                'max_generations': self.params.max_generations
            },
            'convergence_history': {
                'fitness_history': [float(c) for c in self.cost_history],
                'generation_history': self.generation_history,
                'diversity_history': [float(d) for d in self.diversity_history],
                'improvements': [(float(f), int(g)) for f, g in self.best_solutions]
            },
            'resource_utilization': {
                resource: {
                    'average_usage': float(stats['average_usage']),
                    'max_usage': float(stats['max_usage']),
                    'utilization_percent': float(stats['utilization_percent']),
                    'peak_utilization': float(stats['peak_utilization']),
                    'capacity': int(stats['capacity'])
                }
                for resource, stats in utilization_stats.items()
            },
            'resource_efficiency': {
                'overall_utilization': float(
                    sum(stats['utilization_percent']
                        for stats in utilization_stats.values()) / len(utilization_stats)
                ),
                'peak_resource_usage': {
                    resource: float(stats['peak_utilization'])
                    for resource, stats in utilization_stats.items()
                },
                'bottleneck_resource': max(
                    utilization_stats.items(),
                    key=lambda x: x[1]['peak_utilization']
                )[0]
            }
        }

    def _calculate_population_diversity(self, population: List[List[int]]) -> float:
        """Calculate diversity metric for current population"""
        if not population:
            return 0.0

        n = len(population)
        total_distance = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                total_distance += self._solution_distance(population[i], population[j])

        max_possible = self.num_tasks * (n * (n - 1)) / 2
        return total_distance / max_possible if max_possible > 0 else 0.0

    def _solution_distance(self, sol1: List[int], sol2: List[int]) -> float:
        """Calculate normalized distance between two solutions"""
        if not sol1 or not sol2:
            return 0.0

        pos1 = {task: idx for idx, task in enumerate(sol1)}
        pos2 = {task: idx for idx, task in enumerate(sol2)}

        max_distance = len(sol1) * len(sol1) / 2  # Maximum possible position difference
        total_distance = sum(abs(pos1[task] - pos2[task]) for task in range(len(sol1)))

        return total_distance / max_distance

    def _save_report(self, results: Dict):
        """Save detailed analysis report"""
        try:
            report_path = os.path.join(self.output_dir, 'ga_analysis_report.json')

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

            results = convert_to_serializable(results)

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"Analysis report saved to: {report_path}")

        except Exception as e:
            print(f"Error saving report: {str(e)}")
            traceback.print_exc()

    def _initialize_tracking(self):
        """Initialize tracking variables"""
        self.cost_history = []
        self.generation_history = []
        self.diversity_history = []
        self.best_solutions = []
        self.best_solution = None
        self.start_time = None

    def _create_output_directories(self):
        """Create directories for output files"""
        try:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"ga_results_{self.timestamp}"
            self.viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.viz_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error creating directories: {str(e)}")

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

            return max(early_finish)

        except Exception as e:
            print(f"Error calculating critical path: {str(e)}")
            return self.num_tasks

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

    def _calculate_makespan(self, schedule: List[Dict]) -> float:
        """Calculate makespan of final schedule"""
        try:
            if not schedule:
                return float('inf')
            return max(task['end_time'] for task in schedule)
        except Exception as e:
            print(f"Error calculating makespan: {str(e)}")
            return float('inf')

    def _prepare_error_results(self) -> Dict:
        """Prepare error results"""
        execution_time = time.time() - self.start_time if self.start_time else 0

        return {
            'performance_metrics': {
                'makespan': float('inf'),
                'best_fitness': float('inf'),
                'execution_time': float(execution_time),
                'generations': len(self.generation_history),
                'population_size': self.params.population_size if hasattr(self, 'params') else 0,
                'improvements': 0
            },
            'schedule': [],
            'algorithm_parameters': {
                'population_size': getattr(self, 'population_size', 0),
                'mutation_rate': getattr(self, 'mutation_rate', 0),
                'crossover_rate': getattr(self, 'crossover_rate', 0),
                'tournament_size': getattr(self, 'tournament_size', 0),
                'elite_size': getattr(self, 'elite_size', 0),
                'max_generations': getattr(self, 'max_generations', 0)
            },
            'convergence_history': {
                'fitness_history': [],
                'generation_history': [],
                'diversity_history': [],
                'improvements': []
            },
            'error': 'Optimization failed'
        }

    def _create_random_solution(self) -> List[int]:
        """Create a random valid solution using a modified randomized topological sort"""
        # Initialize in-degree counts
        in_degree = {i: 0 for i in range(self.num_tasks)}
        for task_id, task in enumerate(self.tasks):
            for succ in task.get('successors', []):
                in_degree[succ] += 1

        # Track available tasks (those with in-degree 0)
        available = [i for i, degree in in_degree.items() if degree == 0]
        schedule = []

        while available:
            # Randomize task selection among available tasks
            task = random.choice(available)
            schedule.append(task)
            available.remove(task)

            # Update in-degrees and add newly available tasks
            for succ in self.tasks[task].get('successors', []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    # Insert at random position to increase randomness
                    insert_pos = random.randint(0, len(available))
                    available.insert(insert_pos, succ)

        # Validate solution
        if not self._is_valid_schedule(schedule):
            # If invalid, fall back to simple topological sort
            schedule = self._create_topological_solution()

        return schedule

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

    def _calculate_final_schedule(self, schedule: List[int]) -> List[Dict]:
        """Convert schedule to detailed timing information"""
        try:
            if not schedule:
                return []

            final_schedule = []
            task_times = self._calculate_task_times(schedule)

            for task_id in schedule:
                timing = task_times[task_id]
                final_schedule.append({
                    'task_id': task_id,
                    'start_time': float(timing['start']),
                    'end_time': float(timing['end']),
                    'processing_time': float(self.tasks[task_id]['processing_time'])
                })

            # Sort by start time
            final_schedule = sorted(final_schedule, key=lambda x: x['start_time'])

            return final_schedule

        except Exception as e:
            print(f"Error calculating final schedule: {str(e)}")
            return []

def main():
    try:
        # Configure dataset
        dataset_size = "120"  # Choose from: 30, 60, 90, 120
        json_dir = os.path.join('processed_data', f'j{dataset_size}.sm', 'json')
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        if not json_files:
            raise ValueError(f"No JSON files found in {json_dir}")

        dataset_path = os.path.join(json_dir, json_files[0])
        print(f"Using dataset: {dataset_path}")

        # Run optimization
        scheduler = GeneticScheduler(dataset_path)
        results = scheduler.optimize()

        print("\nOptimization completed successfully")
        print(f"Results and visualizations saved in: {scheduler.output_dir}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()