# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from typing import List, Dict
#
#
# def create_algorithm_comparison_plots(results_data: List[Dict], save_path: str = 'algorithm_comparison'):
#     """Create and save comprehensive comparison plots"""
#     # Group data by algorithm and dataset size
#     algorithms = set(r['Algorithm'] for r in results_data)
#     sizes = sorted(set(r['Dataset Size'] for r in results_data))
#
#     # Set figure style
#     plt.figure(figsize=(15, 10))
#
#     # Plot styles
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Default matplotlib colors
#     markers = ['o', 's', 'D', '^']
#
#     # 1. Execution Time vs Dataset Size
#     plt.subplot(2, 1, 1)
#
#     for idx, algorithm in enumerate(algorithms):
#         # Extract data for this algorithm
#         alg_data = [(r['Dataset Size'], r['Execution Time'])
#                     for r in results_data if r['Algorithm'] == algorithm]
#         x_vals, y_vals = zip(*sorted(alg_data))
#
#         plt.plot(x_vals, y_vals,
#                  marker=markers[idx],
#                  linestyle='-',
#                  linewidth=2,
#                  markersize=8,
#                  label=algorithm,
#                  color=colors[idx])
#
#         # Add value labels
#         for x, y in zip(x_vals, y_vals):
#             plt.annotate(f'{y:.2f}s',
#                          (x, y),
#                          textcoords="offset points",
#                          xytext=(0, 10),
#                          ha='center',
#                          va='bottom',
#                          color=colors[idx])
#
#     plt.title('Algorithm Execution Time vs Dataset Size', fontsize=14, pad=20)
#     plt.xlabel('Dataset Size (number of tasks)', fontsize=12)
#     plt.ylabel('Execution Time (seconds)', fontsize=12)
#     plt.grid(True, alpha=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#
#     # 2. Makespan vs Dataset Size
#     plt.subplot(2, 1, 2)
#
#     for idx, algorithm in enumerate(algorithms):
#         # Extract data for this algorithm
#         alg_data = [(r['Dataset Size'], r['Makespan'])
#                     for r in results_data if r['Algorithm'] == algorithm]
#         x_vals, y_vals = zip(*sorted(alg_data))
#
#         plt.plot(x_vals, y_vals,
#                  marker=markers[idx],
#                  linestyle='-',
#                  linewidth=2,
#                  markersize=8,
#                  label=algorithm,
#                  color=colors[idx])
#
#         # Add value labels
#         for x, y in zip(x_vals, y_vals):
#             plt.annotate(f'{y:.1f}',
#                          (x, y),
#                          textcoords="offset points",
#                          xytext=(0, 10),
#                          ha='center',
#                          va='bottom',
#                          color=colors[idx])
#
#     plt.title('Algorithm Makespan vs Dataset Size', fontsize=14, pad=20)
#     plt.xlabel('Dataset Size (number of tasks)', fontsize=12)
#     plt.ylabel('Makespan', fontsize=12)
#     plt.grid(True, alpha=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#
#     # Adjust layout and save
#     plt.tight_layout()
#
#     # Create output directory if it doesn't exist
#     os.makedirs(save_path, exist_ok=True)
#
#     # Save the plot
#     plot_path = os.path.join(save_path, 'algorithm_comparison.png')
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     print(f"\nPlot saved to: {plot_path}")
#
#     # Print summary statistics
#     print("\nPerformance Summary:")
#     print("\nExecution Time (seconds):")
#     print("-" * 60)
#     print(f"{'Algorithm':<20} {'30':<10} {'60':<10} {'90':<10} {'120':<10}")
#     print("-" * 60)
#
#     for algorithm in algorithms:
#         times = {r['Dataset Size']: r['Execution Time']
#                  for r in results_data if r['Algorithm'] == algorithm}
#         print(f"{algorithm:<20} " +
#               " ".join(f"{times.get(size, 'N/A'):<10.2f}" for size in [30, 60, 90, 120]))
#
#     print("\nMakespan:")
#     print("-" * 60)
#     print(f"{'Algorithm':<20} {'30':<10} {'60':<10} {'90':<10} {'120':<10}")
#     print("-" * 60)
#
#     for algorithm in algorithms:
#         makespans = {r['Dataset Size']: r['Makespan']
#                      for r in results_data if r['Algorithm'] == algorithm}
#         print(f"{algorithm:<20} " +
#               " ".join(f"{makespans.get(size, 'N/A'):<10.1f}" for size in [30, 60, 90, 120]))
#
#
# def main():
#     print("Starting algorithm comparison analysis...")
#
#     # Example results data (replace with your actual results)
#     all_results = [
#         {'Algorithm': 'Genetic Algorithm', 'Dataset Size': 30, 'Execution Time': 7.54, 'Makespan': 43.0},
#         {'Algorithm': 'Genetic Algorithm', 'Dataset Size': 60, 'Execution Time': 36.53, 'Makespan': 85.0},
#         {'Algorithm': 'Genetic Algorithm', 'Dataset Size': 90, 'Execution Time': 112.63, 'Makespan': 81.0},
#         {'Algorithm': 'Genetic Algorithm', 'Dataset Size': 120, 'Execution Time': 256.09, 'Makespan': 111.0},
#
#         {'Algorithm': 'Greedy Algorithm', 'Dataset Size': 30, 'Execution Time': 0.02, 'Makespan': 54.0},
#         {'Algorithm': 'Greedy Algorithm', 'Dataset Size': 60, 'Execution Time': 0.07, 'Makespan': 106.0},
#         {'Algorithm': 'Greedy Algorithm', 'Dataset Size': 90, 'Execution Time': 0.18, 'Makespan': 98.0},
#         {'Algorithm': 'Greedy Algorithm', 'Dataset Size': 120, 'Execution Time': 0.42, 'Makespan': 143.0},
#
#         {'Algorithm': 'Optimal Solver', 'Dataset Size': 30, 'Execution Time': 0.09, 'Makespan': 42.0},
#         {'Algorithm': 'Optimal Solver', 'Dataset Size': 60, 'Execution Time': 0.15, 'Makespan': 85.0},
#         {'Algorithm': 'Optimal Solver', 'Dataset Size': 90, 'Execution Time': 0.11, 'Makespan': 77.0},
#         {'Algorithm': 'Optimal Solver', 'Dataset Size': 120, 'Execution Time': 0.14, 'Makespan': 111.0},
#
#         {'Algorithm': 'Simulated Annealing', 'Dataset Size': 30, 'Execution Time': 0.62, 'Makespan': 43.0},
#         {'Algorithm': 'Simulated Annealing', 'Dataset Size': 60, 'Execution Time': 0.41, 'Makespan': 88.0},
#         {'Algorithm': 'Simulated Annealing', 'Dataset Size': 90, 'Execution Time': 85.71, 'Makespan': 80.0},
#         {'Algorithm': 'Simulated Annealing', 'Dataset Size': 120, 'Execution Time': 9.10, 'Makespan': 115.0}
#     ]
#
#     # Create visualization and print summary
#     create_algorithm_comparison_plots(all_results)
#
#
# if __name__ == "__main__":
#     main()

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict


def save_summary_to_file(summary: str, file_path: str):
    """Save text summary to a file."""
    with open(file_path, 'w') as file:
        file.write(summary)
    print(f"Summary saved to: {file_path}")


def create_and_save_execution_time_plot(results_data: List[Dict], save_path: str):
    """Create and save Execution Time vs Dataset Size plot."""
    algorithms = set(r['Algorithm'] for r in results_data)
    sizes = sorted(set(r['Dataset Size'] for r in results_data))

    plt.figure(figsize=(10, 6))

    # Plot styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', 'D', '^']

    for idx, algorithm in enumerate(algorithms):
        # Extract data for this algorithm
        alg_data = [(r['Dataset Size'], r['Execution Time'])
                    for r in results_data if r['Algorithm'] == algorithm]
        x_vals, y_vals = zip(*sorted(alg_data))

        plt.plot(x_vals, y_vals,
                 marker=markers[idx],
                 linestyle='-',
                 linewidth=2,
                 markersize=8,
                 label=algorithm,
                 color=colors[idx])

        # Add value labels
        for x, y in zip(x_vals, y_vals):
            plt.annotate(f'{y:.2f}s',
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         va='bottom',
                         color=colors[idx])

    plt.title('Algorithm Execution Time vs Dataset Size', fontsize=14, pad=20)
    plt.xlabel('Dataset Size (number of tasks)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, 'execution_time_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Execution Time plot saved to: {plot_path}")


def create_and_save_makespan_plot(results_data: List[Dict], save_path: str):
    """Create and save Makespan vs Dataset Size plot."""
    algorithms = set(r['Algorithm'] for r in results_data)
    sizes = sorted(set(r['Dataset Size'] for r in results_data))

    plt.figure(figsize=(10, 6))

    # Plot styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', 'D', '^']

    for idx, algorithm in enumerate(algorithms):
        # Extract data for this algorithm
        alg_data = [(r['Dataset Size'], r['Makespan'])
                    for r in results_data if r['Algorithm'] == algorithm]
        x_vals, y_vals = zip(*sorted(alg_data))

        plt.plot(x_vals, y_vals,
                 marker=markers[idx],
                 linestyle='-',
                 linewidth=2,
                 markersize=8,
                 label=algorithm,
                 color=colors[idx])

        # Add value labels
        for x, y in zip(x_vals, y_vals):
            plt.annotate(f'{y:.1f}',
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         va='bottom',
                         color=colors[idx])

    plt.title('Algorithm Makespan vs Dataset Size', fontsize=14, pad=20)
    plt.xlabel('Dataset Size (number of tasks)', fontsize=12)
    plt.ylabel('Makespan', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, 'makespan_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Makespan plot saved to: {plot_path}")


def create_algorithm_comparison_summary(results_data: List[Dict], save_path: str):
    """Create and save performance summary to a file."""
    algorithms = set(r['Algorithm'] for r in results_data)
    sizes = sorted(set(r['Dataset Size'] for r in results_data))

    summary = "\nPerformance Summary:\n"
    summary += "\nExecution Time (seconds):\n"
    summary += "-" * 60 + "\n"
    summary += f"{'Algorithm':<20} {'30':<10} {'60':<10} {'90':<10} {'120':<10}\n"
    summary += "-" * 60 + "\n"

    for algorithm in algorithms:
        times = {r['Dataset Size']: r['Execution Time']
                 for r in results_data if r['Algorithm'] == algorithm}
        summary += f"{algorithm:<20} " + " ".join(f"{times.get(size, 'N/A'):<10.2f}" for size in [30, 60, 90, 120]) + "\n"

    summary += "\nMakespan:\n"
    summary += "-" * 60 + "\n"
    summary += f"{'Algorithm':<20} {'30':<10} {'60':<10} {'90':<10} {'120':<10}\n"
    summary += "-" * 60 + "\n"

    for algorithm in algorithms:
        makespans = {r['Dataset Size']: r['Makespan']
                     for r in results_data if r['Algorithm'] == algorithm}
        summary += f"{algorithm:<20} " + " ".join(f"{makespans.get(size, 'N/A'):<10.1f}" for size in [30, 60, 90, 120]) + "\n"

    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, 'performance_summary.txt')
    save_summary_to_file(summary, file_path)


def main():
    print("Starting algorithm comparison analysis...")

    # Example results data (replace with your actual results)
    all_results = [
        {'Algorithm': 'Genetic Algorithm', 'Dataset Size': 30, 'Execution Time': 7.54, 'Makespan': 43.0},
        {'Algorithm': 'Genetic Algorithm', 'Dataset Size': 60, 'Execution Time': 36.53, 'Makespan': 85.0},
        {'Algorithm': 'Genetic Algorithm', 'Dataset Size': 90, 'Execution Time': 112.63, 'Makespan': 81.0},
        {'Algorithm': 'Genetic Algorithm', 'Dataset Size': 120, 'Execution Time': 256.09, 'Makespan': 111.0},

        {'Algorithm': 'Greedy Algorithm', 'Dataset Size': 30, 'Execution Time': 0.02, 'Makespan': 54.0},
        {'Algorithm': 'Greedy Algorithm', 'Dataset Size': 60, 'Execution Time': 0.07, 'Makespan': 106.0},
        {'Algorithm': 'Greedy Algorithm', 'Dataset Size': 90, 'Execution Time': 0.18, 'Makespan': 98.0},
        {'Algorithm': 'Greedy Algorithm', 'Dataset Size': 120, 'Execution Time': 0.42, 'Makespan': 143.0},

        {'Algorithm': 'Optimal Solver', 'Dataset Size': 30, 'Execution Time': 0.09, 'Makespan': 42.0},
        {'Algorithm': 'Optimal Solver', 'Dataset Size': 60, 'Execution Time': 0.15, 'Makespan': 85.0},
        {'Algorithm': 'Optimal Solver', 'Dataset Size': 90, 'Execution Time': 0.11, 'Makespan': 77.0},
        {'Algorithm': 'Optimal Solver', 'Dataset Size': 120, 'Execution Time': 0.14, 'Makespan': 111.0},

        {'Algorithm': 'Simulated Annealing', 'Dataset Size': 30, 'Execution Time': 0.62, 'Makespan': 43.0},
        {'Algorithm': 'Simulated Annealing', 'Dataset Size': 60, 'Execution Time': 0.41, 'Makespan': 88.0},
        {'Algorithm': 'Simulated Annealing', 'Dataset Size': 90, 'Execution Time': 85.71, 'Makespan': 80.0},
        {'Algorithm': 'Simulated Annealing', 'Dataset Size': 120, 'Execution Time': 9.10, 'Makespan': 115.0}
    ]

    save_path = 'algorithm_comparison'
    create_and_save_execution_time_plot(all_results, save_path)
    create_and_save_makespan_plot(all_results, save_path)
    create_algorithm_comparison_summary(all_results, save_path)


if __name__ == "__main__":
    main()
