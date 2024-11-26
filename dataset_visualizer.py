import os
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List


class RCPSPDatasetVisualizer:
    def __init__(self, json_dir: str, output_dir: str):
        """Initialize visualizer"""
        self.json_dir = json_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_dataset(self, dataset_size: str) -> List[Dict]:
        """Load all JSON files for a given dataset size"""
        datasets = []
        for filename in os.listdir(self.json_dir):
            if filename.startswith(f"j{dataset_size}") and filename.endswith('.json'):
                with open(os.path.join(self.json_dir, filename), 'r') as f:
                    datasets.append(json.load(f))
        return datasets

    def visualize_task_characteristics(self, dataset: Dict, output_prefix: str):
        """Visualize task characteristics"""
        tasks = dataset['tasks']

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # 1. Processing Times Distribution
        processing_times = [task['processing_time'] for task in tasks]
        ax1.hist(processing_times, bins=20, color='skyblue', edgecolor='black')
        ax1.set_title('Processing Times Distribution')
        ax1.set_xlabel('Processing Time')
        ax1.set_ylabel('Count')

        # 2. Resource Requirements
        resource_usage = {res: [] for res in dataset['dataset_metadata']['global_resources'].keys()}
        for task in tasks:
            for res, amount in task['resource_requirements'].items():
                resource_usage[res].append(amount)

        # Box plot for resource requirements
        ax2.boxplot(resource_usage.values(), tick_labels=resource_usage.keys())
        ax2.set_title('Resource Requirements Distribution')
        ax2.set_ylabel('Amount')

        # 3. Number of Successors
        successor_counts = [len(task['successors']) for task in tasks]
        ax3.hist(successor_counts, bins=range(max(successor_counts) + 2), color='lightgreen', edgecolor='black')
        ax3.set_title('Number of Successors Distribution')
        ax3.set_xlabel('Number of Successors')
        ax3.set_ylabel('Count')

        # 4. Task Network Density
        network_density = []
        resource_limits = dataset['dataset_metadata']['global_resources']
        resource_usage_over_time = {res: [] for res in resource_limits.keys()}

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_task_characteristics.png'))
        plt.close()

    def visualize_precedence_network(self, dataset: Dict, output_prefix: str):
        """Visualize task precedence network"""
        G = nx.DiGraph()

        # Add nodes with attributes
        for task in dataset['tasks']:
            G.add_node(task['task_id'],
                       processing_time=task['processing_time'],
                       resources=task['resource_requirements'])

        # Add edges
        for task in dataset['tasks']:
            for succ in task['successors']:
                G.add_edge(task['task_id'], succ)

        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes
        node_sizes = [1000 if task['processing_time'] > 0 else 500 for task in dataset['tasks']]
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                               node_size=node_sizes)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)

        # Add labels
        labels = {task['task_id']: f"T{task['task_id']}\n({task['processing_time']})"
                  for task in dataset['tasks']}
        nx.draw_networkx_labels(G, pos, labels)

        plt.title('Task Precedence Network\nNode size: proportional to processing time')
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_precedence_network.png'))
        plt.close()

    def visualize_resource_profile(self, dataset: Dict, output_prefix: str):
        """Visualize resource utilization profile"""
        tasks = dataset['tasks']
        resources = dataset['dataset_metadata']['global_resources']

        # Calculate earliest possible start times (naive)
        est = {0: 0}  # task 0 starts at time 0
        for task in sorted(tasks, key=lambda x: x['task_id']):
            task_id = task['task_id']
            est[task_id] = 0
            for pred in [t for t in tasks if task_id in t['successors']]:
                est[task_id] = max(est[task_id],
                                   est[pred['task_id']] + pred['processing_time'])

        # Calculate resource usage over time
        makespan = max(est[task['task_id']] + task['processing_time'] for task in tasks)
        resource_usage = {res: [0] * int(makespan) for res in resources.keys()}

        for task in tasks:
            start = est[task['task_id']]
            for t in range(start, start + task['processing_time']):
                for res, amount in task['resource_requirements'].items():
                    resource_usage[res][t] += amount

        # Plot resource usage
        plt.figure(figsize=(15, 10))
        for res, usage in resource_usage.items():
            plt.plot(usage, label=f"{res} (limit: {resources[res]})")
            plt.axhline(y=resources[res], linestyle='--', alpha=0.5)

        plt.title('Resource Utilization Profile')
        plt.xlabel('Time')
        plt.ylabel('Resource Usage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, f'{output_prefix}_resource_profile.png'))
        plt.close()

    def create_dataset_summary(self, datasets: List[Dict], dataset_size: str):
        """Create summary visualization for the dataset group"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # 1. Processing Time Statistics
        all_times = []
        for dataset in datasets:
            times = [task['processing_time'] for task in dataset['tasks']]
            all_times.extend(times)

        ax1.hist(all_times, bins=30, color='skyblue', edgecolor='black')
        ax1.set_title(f'Processing Times Distribution - {dataset_size} Jobs')
        ax1.set_xlabel('Processing Time')
        ax1.set_ylabel('Count')

        # 2. Resource Usage Statistics
        resource_usage = {f'R{i + 1}': [] for i in range(4)}
        for dataset in datasets:
            for task in dataset['tasks']:
                for res, amount in task['resource_requirements'].items():
                    resource_usage[res].append(amount)

        ax2.boxplot(resource_usage.values(), tick_labels=resource_usage.keys())
        ax2.set_title('Resource Requirements Distribution')
        ax2.set_ylabel('Amount')

        # 3. Network Statistics
        network_stats = []
        for dataset in datasets:
            G = nx.DiGraph()
            for task in dataset['tasks']:
                for succ in task['successors']:
                    G.add_edge(task['task_id'], succ)
            network_stats.append({
                'density': nx.density(G),
                'num_edges': G.number_of_edges()
            })

        densities = [stat['density'] for stat in network_stats]
        ax3.hist(densities, bins=20, color='lightgreen', edgecolor='black')
        ax3.set_title('Network Density Distribution')
        ax3.set_xlabel('Network Density')
        ax3.set_ylabel('Count')

        # 4. Summary Statistics
        stats_text = f"Dataset Size: {dataset_size} jobs\n"
        stats_text += f"Number of Instances: {len(datasets)}\n"
        stats_text += f"Avg Processing Time: {np.mean(all_times):.2f}\n"
        stats_text += f"Max Processing Time: {np.max(all_times):.2f}\n"
        stats_text += f"Avg Network Density: {np.mean(densities):.3f}"

        ax4.text(0.5, 0.5, stats_text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax4.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{dataset_size}_summary.png'))
        plt.close()


def main():
    base_dir = 'processed_data'
    dataset_sizes = ['30', '60', '90', '120']

    for size in dataset_sizes:
        print(f"\nProcessing {size}-job datasets...")

        json_dir = os.path.join(base_dir, f'j{size}.sm', 'json')
        output_dir = os.path.join(base_dir, f'j{size}.sm', 'visualizations')

        if not os.path.exists(json_dir):
            print(f"No data found for {size}-job datasets")
            continue

        visualizer = RCPSPDatasetVisualizer(json_dir, output_dir)
        datasets = visualizer.load_dataset(size)

        if not datasets:
            print(f"No datasets loaded for {size}-job instances")
            continue

        print(f"Creating visualizations for {len(datasets)} instances...")

        # Create summary visualization for the dataset group
        visualizer.create_dataset_summary(datasets, size)

        # Create detailed visualizations for first instance as example
        example_dataset = datasets[0]
        prefix = f"example_{size}"
        visualizer.visualize_task_characteristics(example_dataset, prefix)
        visualizer.visualize_precedence_network(example_dataset, prefix)
        visualizer.visualize_resource_profile(example_dataset, prefix)

        print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()