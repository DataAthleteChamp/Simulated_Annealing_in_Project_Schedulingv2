import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import List, Dict
from genetic_algorithm import GeneticScheduler
from greedy import GreedyScheduler
from optimal_solver import OptimalScheduler
from simulated_annealing import SimulatedAnnealingScheduler


def load_dataset_paths(base_path: str = 'processed_data') -> Dict[str, List[str]]:
    """Load available dataset paths organized by size"""
    dataset_paths = {}
    sizes = ['30', '60', '90', '120']

    for size in sizes:
        json_dir = os.path.join(base_path, f'j{size}.sm', 'json')
        if os.path.exists(json_dir):
            paths = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]
            if paths:
                dataset_paths[f'j{size}'] = paths

    return dataset_paths


def run_algorithm(algorithm: str, dataset_path: str) -> Dict:
    """Run selected scheduling algorithm and return results"""
    schedulers = {
        'Genetic Algorithm': GeneticScheduler,
        'Greedy Algorithm': GreedyScheduler,
        'Optimal Solver': OptimalScheduler,
        'Simulated Annealing': SimulatedAnnealingScheduler
    }

    scheduler = schedulers[algorithm](dataset_path)
    results = scheduler.optimize()

    # Extract key metrics
    metrics = {
        'Algorithm': algorithm,
        'Makespan': results['performance_metrics']['makespan'],
        'Execution Time': results['performance_metrics']['execution_time']
    }


    return metrics

def create_comparison_plots(results_df: pd.DataFrame):
    """Create improved comparison visualizations with values on bars"""
    # Makespan Comparison with adjusted baseline
    fig_makespan = plt.figure(figsize=(12, 6))
    min_makespan = results_df['Makespan'].min()
    # Calculate percentage difference from best makespan
    results_df['Makespan_Difference'] = ((results_df['Makespan'] - min_makespan) / min_makespan) * 100

    # Create subplot for absolute makespan
    ax1 = plt.subplot(1, 2, 1)
    bars1 = sns.barplot(x='Algorithm', y='Makespan', data=results_df, ax=ax1)
    plt.title('Absolute Makespan'
              ' (Lower is Better)')
    plt.ylabel('Makespan')
    plt.xticks(rotation=45)
    # Add labels to bars
    for bar in bars1.containers:
        bars1.bar_label(bar, fmt='%.2f')

    # Create subplot for relative difference
    ax2 = plt.subplot(1, 2, 2)
    bars2 = sns.barplot(x='Algorithm', y='Makespan_Difference', data=results_df, ax=ax2)
    plt.title('Makespan Difference (%)')
    plt.ylabel('Difference from Best (%)')
    plt.xticks(rotation=45)
    # Add labels to bars
    for bar in bars2.containers:
        bars2.bar_label(bar, fmt='%.2f')

    plt.tight_layout()
    st.pyplot(fig_makespan)

    # Execution Time Comparison with log scale
    fig_time = plt.figure(figsize=(12, 6))

    # Create subplot for absolute time
    ax3 = plt.subplot(1, 2, 1)
    bars3 = sns.barplot(x='Algorithm', y='Execution Time', data=results_df, ax=ax3)
    plt.title('Absolute Execution Time (in seconds)')
    plt.ylabel('Execution Time (s)')
    plt.xticks(rotation=45)
    # Add labels to bars
    for bar in bars3.containers:
        bars3.bar_label(bar, fmt='%.2f')

    # Create subplot for log scale time
    ax4 = plt.subplot(1, 2, 2)
    bars4 = sns.barplot(x='Algorithm', y='Execution Time', data=results_df, ax=ax4)
    plt.yscale('log')
    plt.title('Execution Time (Log Scale, in seconds)')
    plt.ylabel('Execution Time (log scale, s)')
    plt.xticks(rotation=45)
    # Add labels to bars
    for bar in bars4.containers:
        bars4.bar_label(bar, fmt='%.2f')

    plt.tight_layout()
    st.pyplot(fig_time)


def main():
    st.title("Project Scheduling Algorithm Comparison")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Algorithm selection
    available_algorithms = [
        'Genetic Algorithm',
        'Greedy Algorithm',
        'Optimal Solver',
        'Simulated Annealing'
    ]
    selected_algorithms = st.sidebar.multiselect(
        "Select Algorithms to Compare",
        available_algorithms,
        default=['Genetic Algorithm', 'Simulated Annealing']
    )

    # Dataset selection
    dataset_paths = load_dataset_paths()
    selected_size = st.sidebar.selectbox(
        "Select Dataset Size",
        list(dataset_paths.keys()),
        format_func=lambda x: f"{x} tasks"
    )

    selected_instance = st.sidebar.selectbox(
        "Select Dataset Instance",
        range(len(dataset_paths[selected_size])),
        format_func=lambda x: f"Instance {x + 1}"
    )

    dataset_path = dataset_paths[selected_size][selected_instance]

    # Run comparison
    if st.button("Run Comparison"):
        if not selected_algorithms:
            st.warning("Please select at least one algorithm to compare.")
            return

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, algorithm in enumerate(selected_algorithms):
            status_text.text(f"Running {algorithm}...")
            results.append(run_algorithm(algorithm, dataset_path))
            progress_bar.progress((i + 1) / len(selected_algorithms))

        status_text.text("Analysis complete!")

        # Display results
        results_df = pd.DataFrame(results)

        st.header("Comparison Results")

        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(results_df.style.highlight_min(['Makespan', 'Execution Time'], color='lightgreen'))

        # Detailed metrics
        st.subheader("Performance Comparison")
        create_comparison_plots(results_df)

if __name__ == "__main__":
    main()