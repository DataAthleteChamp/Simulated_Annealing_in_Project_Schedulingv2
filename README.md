# Simulated Annealing in Project Scheduling

## About The Project
This repository contains the implementation of various optimization algorithms for solving the Resource-Constrained Project Scheduling Problem (RCPSP), with a particular focus on the Simulated Annealing algorithm. The project was developed as part of a bachelor's thesis at Wrocław University of Science and Technology.

### Features
- Implementation of multiple optimization algorithms:
  - Simulated Annealing (SA)
  - Genetic Algorithm (GA)
  - Greedy Algorithm
  - APKA Algorithm implementation
  - Optimal solver integration
- Adaptive temperature scheduling
- Dataset visualization tools
- Support for standard RCPSP datasets
- Performance comparison capabilities

## Getting Started

### Prerequisites
- Python 3.12
- Git (optional)
- 4GB RAM minimum
- Any operating system (Windows/Linux/MacOS)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/DataAthleteChamp/Simulated_Annealing_in_Project_Schedulingv2.git
cd Simulated_Annealing_in_Project_Schedulingv2
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Project Structure
```
├── .idea/                    # IDE configuration files
├── data/                     # Project datasets
├── processed_data/           # Processed dataset files
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation
├── adaptive_temperature_scheduler.py    # Adaptive SA implementation
├── apka.py                  # APKA algorithm implementation
├── dataset_visualizer.py    # Dataset visualization tools
├── genetic_algorithm.py     # Genetic algorithm implementation
├── greedy.py               # Greedy algorithm implementation
├── greedy_scheduler.log    # Logging file for greedy scheduler
├── optimal_solver.py       # Optimal solver implementation
├── rcpsp_parser.py        # RCPSP dataset parser
├── requirements.txt       # Project dependencies
└── simulated_annealing.py # Main SA implementation
```

## Usage

### Running the Algorithms

#### Simulated Annealing
```python
from simulated_annealing import SimulatedAnnealing
sa = SimulatedAnnealing('data/instance.sm')
solution = sa.optimize()
```

#### Using the APKA Algorithm
```python
from apka import APKA
solver = APKA('data/instance.sm')
result = solver.solve()
```

#### Visualizing Results
```python
from dataset_visualizer import DatasetVisualizer
visualizer = DatasetVisualizer('data/instance.sm')
visualizer.plot_schedule(solution)
```

## Implementation Details

### Algorithm Features
- Adaptive temperature control for Simulated Annealing
- Multiple optimization strategies
- RCPSP dataset parsing and validation
- Solution visualization capabilities
- Performance logging and analysis

### Data Processing
- Support for standard RCPSP file formats
- Dataset preprocessing and validation
- Result analysis and comparison tools

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add some NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## Contact
Jakub Piotrowski - [GitHub Profile](https://github.com/DataAthleteChamp)

Project Link: [https://github.com/DataAthleteChamp/Simulated_Annealing_in_Project_Schedulingv2](https://github.com/DataAthleteChamp/Simulated_Annealing_in_Project_Schedulingv2)

## Acknowledgments
- PSPLIB for providing the test datasets
- Wrocław University of Science and Technology
- Dr. Magdalena Turowska for project supervision