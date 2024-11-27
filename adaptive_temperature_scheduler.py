import numpy as np
from typing import Dict


class AdaptiveTemperatureScheduler:
    def __init__(self, initial_temp: float, min_temp: float, target_acceptance_rate: float = 0.3):
        """Initialize the adaptive temperature scheduler.

        Args:
            initial_temp: Initial temperature value
            min_temp: Minimum temperature value
            target_acceptance_rate: Target solution acceptance rate (default: 0.3)
        """
        self.initial_temp = initial_temp  # Add this line to initialize the attribute
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.target_acceptance_rate = target_acceptance_rate
        self.acceptance_history = []
        self.cooling_rate = 0.95
        self.window_size = 100


    def calculate_cooling_rate(self, current_cost: float, best_cost: float, iteration: int) -> float:
        """Calculate adaptive cooling rate based on search progress."""
        if len(self.acceptance_history) >= self.window_size:
            current_acceptance = sum(self.acceptance_history[-self.window_size:]) / self.window_size

            if current_acceptance > self.target_acceptance_rate * 1.2:
                self.cooling_rate = max(0.90, self.cooling_rate * 0.95)
            elif current_acceptance < self.target_acceptance_rate * 0.8:
                self.cooling_rate = min(0.99, self.cooling_rate * 1.05)
            else:
                self.cooling_rate = 0.95

        # Adjust based on solution quality
        quality_ratio = current_cost / best_cost if best_cost > 0 else 1.0
        if quality_ratio > 1.2:  # Far from best
            self.cooling_rate = max(0.90, self.cooling_rate * 0.98)

        return self.cooling_rate

    def get_next_temperature(self, current_cost: float, best_cost: float, iteration: int, was_accepted: bool) -> float:
        """Calculate next temperature value."""
        # Track acceptance
        self.acceptance_history.append(1 if was_accepted else 0)

        # Get adaptive cooling rate
        cooling_rate = self.calculate_cooling_rate(current_cost, best_cost, iteration)

        # Calculate new temperature
        new_temp = max(self.min_temp, self.current_temp * cooling_rate)

        # Calculate recent acceptance rate
        if len(self.acceptance_history) >= self.window_size:
            recent_acceptance = sum(self.acceptance_history[-self.window_size:]) / self.window_size
        else:
            recent_acceptance = 0  # Default value if not enough history

        # Implement reheating if temperature stagnates
        if (self.current_temp <= self.min_temp and
                recent_acceptance >= self.target_acceptance_rate * 0.9 and
                iteration - getattr(self, 'last_reheat_iteration', -float('inf')) > 100):
            new_temp = min(self.initial_temp, self.current_temp * 1.5)  # Reheat temperature
            self.last_reheat_iteration = iteration  # Update last reheating iteration
            print(f"Reheating applied: New Temperature = {new_temp:.2f}")

        # Update the current temperature
        self.current_temp = new_temp

        # Debugging: Monitor temperature, cooling rate, and acceptance rate
        # if iteration % 500 == 0:
        #     print(f"Cooling Rate: {cooling_rate:.4f}, Recent Acceptance: {recent_acceptance:.2f}")
        return self.current_temp

    def _should_reheat(self, iteration: int) -> bool:
        """Determine if reheating should be applied."""
        if len(self.acceptance_history) < self.window_size:
            return False

        recent_acceptance = sum(self.acceptance_history[-self.window_size:]) / self.window_size
        return recent_acceptance < self.target_acceptance_rate * 0.5

    def _apply_reheating(self, current_temp: float) -> float:
        """Apply reheating when search is stuck."""
        reheat_factor = 2.0
        self.acceptance_history = []  # Reset acceptance history after reheating
        return current_temp * reheat_factor

    def get_statistics(self) -> Dict:
        """Return current temperature statistics."""
        recent_acceptance = (sum(self.acceptance_history[-self.window_size:]) / self.window_size
                             if len(self.acceptance_history) >= self.window_size else 0)
        return {
            'current_temperature': self.current_temp,
            'cooling_rate': self.cooling_rate,
            'recent_acceptance_rate': recent_acceptance,
        }