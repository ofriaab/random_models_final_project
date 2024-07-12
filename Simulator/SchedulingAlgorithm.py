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
        min_task=min(ready_tasks, key=lambda task: task.runtime)
        if min_task is not None:
            ready_tasks.remove(min_task)
        return min_task

class MaxOutdegreeAlgorithm(SchedulingAlgorithm):
    def choose_next_task(self, ready_tasks):
        max_task= max(ready_tasks, key=lambda task: task.out_degree)
        if max_task is not None:
                ready_tasks.remove(max_task)
        return max_task