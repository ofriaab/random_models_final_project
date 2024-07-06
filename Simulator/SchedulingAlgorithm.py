from abc import ABC, abstractmethod

class SchedulingAlgorithm(ABC):
    @abstractmethod
    def choose_next_task(self, ready_tasks):
        pass

class SimpleQueueAlgorithm(SchedulingAlgorithm):
    def choose_next_task(self, ready_tasks):
        return ready_tasks.pop()  # FIFO

class MinimalRuntimeAlgorithm(SchedulingAlgorithm):
    def choose_next_task(self, ready_tasks):
        min_task = None
        min_runtime = float('inf')  # Initialize with a very large number

        for task in ready_tasks:
            current_runtime = task.runtime
            if current_runtime < min_runtime:
                min_runtime = current_runtime
                min_task = task

        if min_task is not None:
            ready_tasks.remove(min_task)

        return min_task

class MaxOutdegreeAlgorithm(SchedulingAlgorithm):
    def choose_next_task(self, ready_tasks):
        max_task = None
        max_outdegree = -1

        for task in ready_tasks:
            current_outdegree = len(task.dependencies)
            if current_outdegree > max_outdegree:
                max_outdegree = current_outdegree
                max_task = task

        if max_task is not None:
            ready_tasks.remove(max_task)

        return max_task
