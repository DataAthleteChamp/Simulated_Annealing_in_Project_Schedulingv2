import os
from typing import Dict, List
import json


class RCPSPParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._parse_file()

    def _parse_file(self) -> Dict:
        """Parse RCPSP file format"""
        try:
            with open(self.file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]

            # Find number of jobs and resources
            num_jobs = None
            num_resources = None
            for line in lines:
                if 'jobs (incl. supersource/sink )' in line:
                    num_jobs = int(line.split(':')[1].strip().split()[0])
                elif 'renewable' in line:
                    num_resources = int(line.split(':')[1].strip().split()[0])
                    break

            if not num_jobs or not num_resources:
                raise ValueError("Could not find number of jobs or resources")

            # Find resource availabilities (at the end of file)
            resource_capacities = None
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if 'RESOURCEAVAILABILITIES:' in line:
                    resource_capacities = [int(x) for x in lines[i + 2].strip().split()]
                    break

            if not resource_capacities:
                raise ValueError("Could not find resource capacities")

            # Create resources dictionary
            resources = {f'R{i + 1}': capacity
                         for i, capacity in enumerate(resource_capacities)}

            # Find precedence relations section
            precedence_start = None
            for i, line in enumerate(lines):
                if 'PRECEDENCE RELATIONS:' in line:
                    precedence_start = i + 2  # Skip header line
                    break

            if precedence_start is None:
                raise ValueError("Could not find precedence relations")

            # First pass: Get successors for each job
            successors_dict = {}
            current_line = precedence_start
            while current_line < len(lines):
                line = lines[current_line].strip()
                if 'REQUESTS/DURATIONS:' in line:
                    break
                if line and not line.startswith('*'):
                    parts = line.split()
                    if len(parts) >= 3 and parts[0].isdigit():
                        job_id = int(parts[0])
                        num_successors = int(parts[2])
                        successors = [int(x) for x in parts[3:3 + num_successors]]
                        successors_dict[job_id] = successors
                current_line += 1

            # Find durations section
            durations_start = None
            for i, line in enumerate(lines):
                if 'REQUESTS/DURATIONS:' in line:
                    durations_start = i + 3  # Skip header and dashes
                    break

            if durations_start is None:
                raise ValueError("Could not find durations section")

            # Parse jobs
            jobs = []
            current_line = durations_start
            while current_line < len(lines):
                line = lines[current_line].strip()
                if line.startswith('***'):
                    break
                if line and not line.startswith('-'):
                    parts = line.split()
                    if len(parts) >= 2 + num_resources and parts[0].isdigit():
                        job_id = int(parts[0])
                        duration = int(parts[2])
                        resource_reqs = {
                            f'R{i + 1}': int(parts[i + 3])
                            for i in range(num_resources)
                        }

                        job = {
                            'task_id': job_id - 1,  # Convert to 0-based indexing
                            'processing_time': duration,
                            'resource_requirements': resource_reqs,
                            'successors': [x - 1 for x in successors_dict.get(job_id, [])],  # Convert to 0-based
                            'priority': 1.0
                        }
                        jobs.append(job)
                current_line += 1

            if not jobs:
                raise ValueError("No jobs were parsed")

            # Create final dataset
            dataset = {
                'dataset_metadata': {
                    'num_jobs': num_jobs,
                    'num_resources': num_resources,
                    'global_resources': resources,
                    'file_name': os.path.basename(self.file_path)
                },
                'tasks': sorted(jobs, key=lambda x: x['task_id'])  # Sort by task_id
            }

            return dataset

        except Exception as e:
            print(f"Error parsing file {self.file_path}: {str(e)}")
            return None

    def save_json(self, output_path: str):
        """Save parsed data as JSON"""
        if self.data:
            with open(output_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            print(f"Saved JSON to {output_path}")


def process_instance_set(instance_dir: str, output_dir: str):
    """Process all instances in a directory"""
    print(f"\nProcessing directory: {instance_dir}")
    os.makedirs(output_dir, exist_ok=True)
    json_dir = os.path.join(output_dir, 'json')
    os.makedirs(json_dir, exist_ok=True)

    for filename in os.listdir(instance_dir):
        if filename.endswith('.sm'):
            print(f"\nProcessing {filename}...")

            try:
                file_path = os.path.join(instance_dir, filename)
                parser = RCPSPParser(file_path)

                if parser.data and len(parser.data['tasks']) > 0:
                    json_path = os.path.join(json_dir, f"{filename[:-3]}.json")
                    parser.save_json(json_path)
                    print(f"Successfully processed {filename} - {len(parser.data['tasks'])} tasks")
                else:
                    print(f"Failed to parse {filename} or no tasks found")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


def main():
    # Process each instance set
    for instance_set in ['j30.sm', 'j60.sm', 'j90.sm', 'j120.sm']:
        print(f"\nProcessing {instance_set} dataset...")
        instance_dir = os.path.join('data', instance_set)
        output_dir = os.path.join('processed_data', instance_set)
        process_instance_set(instance_dir, output_dir)


if __name__ == "__main__":
    main()