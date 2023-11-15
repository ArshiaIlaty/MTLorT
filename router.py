
import random
from typing import List, Tuple

class Router:
    """
    A class that samples data from multiple tasks according to their weights.

    Attributes:
    -----------
    data : List[Tuple]
        A list of tuples, where each tuple contains the data for a single task.
    task_weights : List[float]
        A list of weights for each task. The length of this list should be equal to the number of tasks.
    
    Methods:
    --------
    sample(batch_size: int) -> Tuple[List, List]:
        Samples data from the tasks according to their weights.
        Returns a tuple containing a list of data for each task and a list of indices indicating which task each data point belongs to.
    """
    def __init__(self, data: List[Tuple], task_weights: List[float]):
        self.data = data
        self.task_weights = task_weights
        self.task_probs = self._compute_task_probs()

    def _compute_task_probs(self) -> List[float]:
        """
        Computes the probability of sampling from each task based on their weights.

        Returns:
        --------
        List[float]:
            A list of probabilities, where the i-th element is the probability of sampling from the i-th task.
        """
        total_weight = sum(self.task_weights)
        return [weight / total_weight for weight in self.task_weights]

    def sample(self, batch_size: int) -> Tuple[List, List]:
        """
        Samples data from the tasks according to their weights.

        Parameters:
        -----------
        batch_size : int
            The number of data points to sample.

        Returns:
        --------
        Tuple[List, List]:
            A tuple containing two lists:
            - A list of length equal to the number of tasks, where each element is a list of data points for that task.
            - A list of length batch_size, where each element is an integer indicating which task the corresponding data point belongs to.
        """
        batch = random.sample(self.data, batch_size)
        task_indices = random.choices(range(len(self.task_probs)), weights=self.task_probs, k=batch_size)
        task_data = [[] for _ in range(len(self.task_probs))]
        for data, task_index in zip(batch, task_indices):
            task_data[task_index].append(data)
        return task_data, task_indices

