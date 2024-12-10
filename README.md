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

2. Create and activate virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```
### Required Dependencies
```
numpy>=1.24.3
pandas>=2.0.2
matplotlib>=3.7.1
streamlit>=1.22.0
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
├── optimal_solver.py       # Optimal solver implementation
├── rcpsp_parser.py        # RCPSP dataset parser
├── requirements.txt       # Project dependencies
└── simulated_annealing.py # Main SA implementation
```

## Running the Project

### 1. Data Preprocessing
To convert raw RCPSP data into processable format, run:
```bash
python rcpsp_parser.py
```
This will process the raw data files from `data/` directory and save processed files in `processed_data/`.

### 2. Running Optimization Algorithms
You can run any of the four optimization algorithms. To test the implementation, it's sufficient to run just one of them:

#### Simulated Annealing
```python
python simulated_annealing.py
```

#### Genetic Algorithm
```python
python genetic_algorithm.py
```

#### Greedy Algorithm
```python
python greedy.py
```

#### Optimal Solver
```python
python optimal_solver.py
```

To change the dataset being used, modify the dataset path in the respective algorithm file:
```python
# Example in simulated_annealing.py
DATASET_PATH = 'processed_data/j30.sm'  # Change this path to use different dataset
```

### 3. Running the Streamlit Application
To launch the interactive visualization and comparison tool:
```bash
streamlit run apka.py
```
The application will be available at http://localhost:8501

## Implementation Details

### Changing Dataset Size
Available dataset sizes:
- j30 (30 tasks)
- j60 (60 tasks)
- j90 (90 tasks)
- j120 (120 tasks)

To change the dataset size, modify the respective file paths in the algorithm implementations.

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

## Common Issues and Solutions

1. Import errors:
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/MacOS
set PYTHONPATH=%PYTHONPATH%;%cd%          # Windows
```

2. Memory issues with large datasets:
- Reduce population size for genetic algorithm
- Use smaller dataset chunks
- Increase system swap space

3. Streamlit issues:
- Ensure all dependencies are installed correctly
- Check if port 8501 is available
- Try running with `--server.port` option if default port is occupied

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