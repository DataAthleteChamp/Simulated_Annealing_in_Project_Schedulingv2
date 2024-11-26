import numpy as np
import matplotlib.pyplot as plt
import json
import random
import time
from typing import List, Dict, Tuple
import os
from datetime import datetime
import traceback

class GeneticAlgorithmScheduler:
    def __init__(self, dataset_path: str):
        """Initialize the Genetic Algorithm scheduler with dataset"""
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
            if self.num_tasks < 31:  # j30
                population_size = 100
                generations = 200
                elite_size = 10
            elif self.num_tasks < 61:  # j60
                population_size = 150
                generations = 300
                elite_size = 15
            elif self.num_tasks < 91:  # j90
                population_size = 200
                generations = 400
                elite_size = 20
            else:  # j120
                population_size = 250
                generations = 500
                elite_size = 25

            return {
                'population_size': population_size,
                'generations': generations,
                'elite_size': elite_size,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1
            }

        except Exception as e:
            print(f"Error in parameter tuning: {str(e)}")
            return self._get_default_parameters()

    def _get_default_parameters(self) -> Dict:
        """Provide safe default parameters if tuning fails"""
        return {
            'population_size': 200,
            'generations': 500,
            'elite_size': 20,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1
        }

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
            self.output_dir = f"ga_results_{self.timestamp}"
            self.viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.viz_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error creating directories: {str(e)}")

    def _initialize_tracking(self):
        """Initialize tracking variables for optimization"""
        self.params = self._tune_parameters()
        self.best_schedule = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.diversity_metric = []
        self.start_time = None

    def optimize(self) -> Dict:
        """Run the Genetic Algorithm optimization"""
        print("\nStarting optimization process...")
        self.start_time = time.time()

        try:
            population = self._initialize_population()
            self.best_schedule, self.best_cost = self._find_best_solution(population)

            print(f"Initial best makespan: {self.best_cost:.2f}")

            for gen in range(self.params['generations']):
                selected_parents = self._selection(population)
                offspring = self._crossover(selected_parents)
                mutated_offspring = self._mutation(offspring)
                new_population = self._elitism(population, mutated_offspring)

                best_solution, best_cost = self._find_best_solution(new_population)
                if best_cost < self.best_cost:
                    self.best_schedule = best_solution
                    self.best_cost = best_cost
                    print(f"New best solution in generation {gen}: {self.best_cost:.2f}")

                population = new_population
                self._update_tracking(population, gen)

                if self.best_cost <= self._calculate_critical_path_length() * 1.05:
                    print("\nReached near-optimal solution. Stopping early.")
                    break

            execution_time = time.time() - self.start_time
            if self.best_schedule is None:
                print("No valid solution found during optimization.")
                return self._prepare_error_results()
            final_schedule = self._calculate_final_schedule()
            makespan = self._calculate_makespan(final_schedule)

            print("\nOptimization Complete:")
            print(f"Best schedule: {self.best_schedule}")
            print(f"Best cost: {self.best_cost:.2f}")
            print(f"Final makespan: {makespan:.2f}")
            print(f"Total generations: {gen}")
            print(f"Execution time: {execution_time:.2f} seconds")

            results = self._prepare_results(final_schedule)
            self._save_report(results)
            self.create_visualizations()

            return results

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            traceback.print_exc()
            return self._prepare_error_results()

    def _initialize_population(self) -> List[List[int]]:
        """Initialize a population of random solutions"""
        population = []
        for _ in range(self.params['population_size']):
            solution = list(range(self.num_tasks))
            random.shuffle(solution)
            population.append(solution)
        return population

    def _selection(self, population: List[List[int]]) -> List[List[int]]:
        """Select parents using tournament selection"""
        selected_parents = []
        for _ in range(self.params['population_size']):
            candidates = random.sample(population, 2)
            candidate_costs = [self._calculate_cost(c)[0] for c in candidates]
            selected_parents.append(candidates[np.argmin(candidate_costs)])
        return selected_parents

    def _crossover(self, parents: List[List[int]]) -> List[List[int]]:
        """Perform order crossover (OX) to create offspring"""
        offspring = []
        for i in range(0, len(parents), 2):
            if random.random() < self.params['crossover_rate']:
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = self._order_crossover(parent1, parent2)

                # Check validity of offspring
                if self._is_valid_schedule(child1) and self._is_valid_schedule(child2):
                    offspring.append(child1)
                    offspring.append(child2)
                else:
                    offspring.append(parent1)
                    offspring.append(parent2)
            else:
                offspring.append(parents[i].copy())
                offspring.append(parents[i + 1].copy())
        return offspring

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform order crossover between two parents while maintaining precedence constraints"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))

        child1 = [None] * size
        child2 = [None] * size

        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        remaining1 = [t for t in parent2 if t not in child1]
        remaining2 = [t for t in parent1 if t not in child2]

        # Create adjacency lists and in-degree counts for remaining tasks
        adj_list1 = {t: [] for t in remaining1}
        in_degree1 = {t: 0 for t in remaining1}
        adj_list2 = {t: [] for t in remaining2}
        in_degree2 = {t: 0 for t in remaining2}

        for task in remaining1:
            for succ in self.tasks[task].get('successors', []):
                if succ in remaining1:
                    adj_list1[task].append(succ)
                    in_degree1[succ] += 1

        for task in remaining2:
            for succ in self.tasks[task].get('successors', []):
                if succ in remaining2:
                    adj_list2[task].append(succ)
                    in_degree2[succ] += 1

        # Topological sorting to fill remaining positions
        def fill_positions(child, remaining, adj_list, in_degree):
            available = [t for t in remaining if in_degree[t] == 0]
            for i in range(size):
                if child[i] is None:
                    if available:
                        task = available.pop(0)
                        child[i] = task
                        for succ in adj_list[task]:
                            in_degree[succ] -= 1
                            if in_degree[succ] == 0:
                                available.append(succ)
                    else:
                        task = remaining[0]
                        child[i] = task
                        remaining.pop(0)

        fill_positions(child1, remaining1, adj_list1, in_degree1)
        fill_positions(child2, remaining2, adj_list2, in_degree2)

        return child1, child2

    def _mutation(self, offspring: List[List[int]]) -> List[List[int]]:
        """Perform mutation on offspring while maintaining precedence constraints"""
        mutated_offspring = []
        for child in offspring:
            if random.random() < self.params['mutation_rate']:
                valid_mutation = False
                attempts = 0
                while not valid_mutation and attempts < 10:
                    mutated_child = child.copy()
                    task_index = random.randint(0, len(mutated_child) - 1)
                    task = mutated_child.pop(task_index)

                    # Find a valid position to insert the task
                    valid_positions = []
                    for i in range(len(mutated_child)):
                        mutated_child.insert(i, task)
                        if self._is_valid_schedule(mutated_child):
                            valid_positions.append(i)
                        mutated_child.pop(i)

                    if valid_positions:
                        insert_position = random.choice(valid_positions)
                        mutated_child.insert(insert_position, task)
                        valid_mutation = True
                    else:
                        mutated_child = child.copy()

                    attempts += 1

                if valid_mutation:
                    mutated_offspring.append(mutated_child)
                else:
                    mutated_offspring.append(child)
            else:
                mutated_offspring.append(child)
        return mutated_offspring

    def _elitism(self, population: List[List[int]], offspring: List[List[int]]) -> List[List[int]]:
        """Combine current population with offspring, keeping the best solutions"""
        elite_size = self.params['elite_size']
        combined = population + offspring
        costs = [self._calculate_cost(s)[0] for s in combined]
        elite_indices = np.argsort(costs)[:elite_size]
        new_population = [combined[i] for i in elite_indices]

        remaining = [s for i, s in enumerate(combined) if i not in elite_indices]
        new_population.extend(random.sample(remaining, self.params['population_size'] - elite_size))

        return new_population

    def _find_best_solution(self, population: List[List[int]]) -> Tuple[List[int], float]:
        """Find the best solution in the current population"""
        best_solution = None
        best_cost = float('inf')

        for solution in population:
            cost, violations = self._calculate_cost(solution)
            if cost < best_cost:
                best_solution = solution
                best_cost = cost

        if best_solution is None:
            print("No valid solution found in the current population.")
        else:
            print(f"Best solution found: {best_solution}")
            print(f"Best cost: {best_cost:.2f}")

        return best_solution, best_cost

    def _calculate_cost(self, schedule: List[int]) -> Tuple[float, Dict]:
        """Calculate cost ensuring zero violations"""
        try:
            if not self._is_valid_schedule(schedule):
                return float('inf'), {'precedence': 1, 'resource': 1}

            task_times = self._calculate_task_times(schedule)
            makespan = max(timing['end'] for timing in task_times.values())
            return float(makespan), {'precedence': 0, 'resource': 0}

        except Exception as e:
            print(f"Error calculating cost: {str(e)}")
            traceback.print_exc()
            return float('inf'), {'precedence': 1, 'resource': 1}

    def _is_valid_schedule(self, schedule: List[int]) -> bool:
        """Validate schedule feasibility"""
        try:
            if len(set(schedule)) != self.num_tasks:
                print("Invalid schedule: Duplicate or missing tasks")
                return False

            task_positions = {task_id: pos for pos, task_id in enumerate(schedule)}
            for task_id in schedule:
                task = self.tasks[task_id]
                for succ in task.get('successors', []):
                    if task_positions[succ] < task_positions[task_id]:
                       # print(f"Invalid schedule: Precedence constraint violated for task {task_id}")
                        return False

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
                            print(f"Invalid schedule: Resource constraint violated for task {task_id}")
                            return False

            return True

        except Exception as e:
            print(f"Error validating schedule: {str(e)}")
            traceback.print_exc()
            return False

    def _calculate_task_times(self, schedule: List[int]) -> Dict:
        """Calculate task timings considering resource constraints"""
        task_times = {}
        resource_usage = {}

        for task_id in schedule:
            task = self.tasks[task_id]
            start_time = 0

            for dep_id in schedule[:schedule.index(task_id)]:
                if task_id in self.tasks[dep_id].get('successors', []):
                    if dep_id in task_times:
                        start_time = max(start_time, task_times[dep_id]['end'])

            while True:
                can_start = True
                end_time = start_time + task['processing_time']

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

    def _calculate_resource_complexity(self) -> float:
        """Calculate resource complexity considering utilization and variability"""
        try:
            total_complexity = 0
            num_resources = len(self.global_resources)

            for resource, capacity in self.global_resources.items():
                total_demand = sum(task['resource_requirements'][resource]
                                   for task in self.tasks)
                avg_utilization = total_demand / (capacity * self.num_tasks)

                demands = [task['resource_requirements'][resource] for task in self.tasks]
                demand_std = np.std(demands) if len(demands) > 1 else 0
                demand_variability = demand_std / capacity if capacity > 0 else 0

                resource_complexity = (avg_utilization + demand_variability) / 2
                total_complexity += resource_complexity

            return total_complexity / num_resources if num_resources > 0 else 0

        except Exception as e:
            print(f"Error calculating resource complexity: {str(e)}")
            return 0.5

    def _calculate_critical_path_length(self) -> int:
        """Calculate critical path length using forward and backward pass"""
        try:
            early_start = [0] * self.num_tasks
            early_finish = [0] * self.num_tasks

            for i in range(self.num_tasks):
                max_pred_finish = 0
                for j in range(i):
                    if i in self.tasks[j].get('successors', []):
                        max_pred_finish = max(max_pred_finish, early_finish[j])
                    early_start[i] = max_pred_finish
                    early_finish[i] = early_start[i] + self.tasks[i]['processing_time']

                late_start = [max(early_finish)] * self.num_tasks
                late_finish = [max(early_finish)] * self.num_tasks

                for i in range(self.num_tasks - 1, -1, -1):
                    min_succ_start = late_finish[i]
                    for succ in self.tasks[i].get('successors', []):
                        min_succ_start = min(min_succ_start, late_start[succ])
                    late_finish[i] = min_succ_start
                    late_start[i] = late_finish[i] - self.tasks[i]['processing_time']

                return max(early_finish)

        except Exception as e:
            print(f"Error calculating critical path: {str(e)}")
            return self.num_tasks


    def _prepare_results(self, final_schedule: List[Dict]) -> Dict:
            """Prepare results for reporting"""
            execution_time = time.time() - self.start_time
            makespan = self._calculate_makespan(final_schedule)

            return {
                'performance_metrics': {
                    'makespan': float(makespan),
                    'best_cost': float(self.best_cost),
                    'execution_time': float(execution_time),
                    'generations': len(self.cost_history),
                    'violations': {'precedence': 0, 'resource': 0}
                },
                'schedule': final_schedule,
                'algorithm_parameters': self.params,
                'convergence_history': {
                    'costs': [float(c) for c in self.cost_history],
                    'diversity': [float(d) for d in self.diversity_metric]
                }
            }

    def _prepare_error_results(self) -> Dict:
        """Prepare error results for reporting"""
        execution_time = time.time() - self.start_time if self.start_time else 0

        return {
            'performance_metrics': {
                'makespan': float('inf'),
                'best_cost': float('inf'),
                'execution_time': float(execution_time),
                'generations': len(self.cost_history),
                'violations': {'precedence': 0, 'resource': 0}
            },
            'schedule': [],
            'algorithm_parameters': self.params,
            'convergence_history': {
                'costs': [float(c) for c in self.cost_history],
                'diversity': [float(d) for d in self.diversity_metric]
            },
            'error': 'Optimization failed'
        }

    def _save_report(self, result: Dict):
        """Save the analysis report"""
        try:
            report_path = os.path.join(self.output_dir, 'analysis_report.json')

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

    def _update_tracking(self, population: List[List[int]], generation: int):
        """Update tracking metrics during optimization"""
        try:
            self.cost_history.append(float(self.best_cost))
            diversity = self._calculate_diversity(population)
            self.diversity_metric.append(diversity)
        except Exception as e:
            print(f"Error updating tracking: {str(e)}")

    def _calculate_diversity(self, population: List[List[int]]) -> float:
        """Calculate diversity metric for a population"""
        try:
            if not population:
                return 0.0

            n = len(population)
            total_distance = 0.0

            for i in range(n):
                for j in range(i + 1, n):
                    total_distance += self._solution_distance(population[i], population[j])

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

    def create_visualizations(self):
        """Generate all visualizations"""
        if not self.best_schedule:
            print("No schedule to visualize")
            return

        try:
            self._plot_convergence()
            self._plot_schedule()
            self._plot_resource_utilization()
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
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        #plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Diversity history
        plt.subplot(2, 1, 2)
        plt.plot(self.diversity_metric, 'r-', label='Diversity')
        plt.title('Population Diversity Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'convergence.png'))
        plt.close()

    def _plot_schedule(self):
        """Create Gantt chart of the schedule"""
        final_schedule = self._calculate_final_schedule()

        plt.figure(figsize=(15, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.global_resources)))
        resource_colors = dict(zip(self.global_resources.keys(), colors))

        for task in final_schedule:
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

    def _plot_resource_utilization(self):
        """Plot resource utilization over time"""
        final_schedule = self._calculate_final_schedule()
        makespan = self._calculate_makespan(final_schedule)

        timeline = {t: {r: 0 for r in self.global_resources}
                    for t in range(int(makespan) + 1)}

        for task in final_schedule:
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
        dataset_size = "30"
        json_dir = os.path.join('processed_data', f'j{dataset_size}.sm', 'json')
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        if not json_files:
            raise ValueError(f"No JSON files found in {json_dir}")

        dataset_path = os.path.join(json_dir, json_files[0])
        print(f"Using dataset: {dataset_path}")

        scheduler = GeneticAlgorithmScheduler(dataset_path)
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
