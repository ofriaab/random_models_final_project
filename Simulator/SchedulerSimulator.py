from collections import deque
from Task import Task
# from SchedulingAlgorithm import SimpleQueueAlgorithm,MinimalRuntimeAlgorithm,MaxOutdegreeAlgorithm


class SchedulerSimulator:
    """
    A class to simulate scheduling of tasks on multiple processors using different scheduling algorithms.

    Attributes:
        tasks (dict): A dictionary of task_id to Task objects.
        num_processors (int): The number of processors available.
        scheduling_algorithm (SchedulingAlgorithm): The algorithm used to choose the next task to schedule.
        ready_tasks (deque): A deque of tasks that are ready to be scheduled.
        running_tasks (list): A list of tasks that are currently running.
        indegree (dict): A dictionary of task_id to indegree count (number of dependencies).
        current_time (int): The current simulation time.
        processors (list): A list of processors, each holding the task assigned to it or None.
        stats (dict): A dictionary to store various statistics of the simulation.
    """

    def __init__(self, tasks, num_processors, scheduling_algorithm):
        """
        Initializes the SchedulerSimulator with the given tasks, number of processors, and scheduling algorithm.

        Args:
            tasks (list of tuples): A list of tasks where each task is represented as a tuple (task_id, runtime, dependencies).
            num_processors (int): The number of processors available.
            scheduling_algorithm (SchedulingAlgorithm): The algorithm used to choose the next task to schedule.
        """
        self.tasks = {task_id: Task(task_id, runtime, dependencies) for task_id, runtime, dependencies in tasks}
        self.num_processors = num_processors
        self.scheduling_algorithm = scheduling_algorithm
        self.ready_tasks = deque()
        self.running_tasks = []
        self.indegree = {task_id: 0 for task_id in self.tasks}
        self.current_time = 0
        self.processors = [None] * num_processors
        self.stats = {
            "makespan": 0,
            "processor_utilization": 0,
            "throughput": 0,
            "avg_waiting_time": 0,
            "avg_turnaround_time": 0
        }

    def initialize_indegree_and_ready_tasks(self):
        """
        Initializes the indegree for each task and the ready tasks deque.
        """
        for task in self.tasks.values():
            self.indegree[task.task_id]+=len(task.dependencies)
        for task_id, degree in self.indegree.items():
            if degree == 0:
                self.ready_tasks.append(self.tasks[task_id])

    def assign_tasks(self):
        """
        Assigns tasks from the ready queue to available processors based on the scheduling algorithm.
        """
        while self.ready_tasks and any(p is None for p in self.processors):
            task = self.scheduling_algorithm.choose_next_task(self.ready_tasks)
            for i, processor in enumerate(self.processors):
                if processor is None:
                    task.processor = i
                    task.end_time = self.current_time + task.runtime
                    self.running_tasks.append(task)
                    self.processors[i] = task
                    print(f"Time {self.current_time}: Assigned {task.task_id} to processor {i} (runtime: {task.runtime})")
                    break

    def process_next_completion(self):
        """
        Processes the next task completion and updates the ready tasks and processors accordingly.
        """
        self.running_tasks.sort(key=lambda task: task.end_time)
        next_task = self.running_tasks.pop(0)
        self.current_time = next_task.end_time
        self.processors[next_task.processor] = None
        print(f"Time {self.current_time}: Completed {next_task.task_id} on processor {next_task.processor}")
        for task in self.tasks.values():
            if next_task.task_id in task.dependencies:
                print(f'next task id: {next_task.task_id}')
                task.dependencies.remove(next_task.task_id)
                if not task.dependencies:
                    self.ready_tasks.append(task)

    def run(self):
        """
        Runs the scheduling simulation until all tasks are completed.
        """
        self.initialize_indegree_and_ready_tasks()
        while self.running_tasks or self.ready_tasks:
            self.assign_tasks()
            if self.running_tasks:
                self.process_next_completion()


    def save_statistics(self,file_name='statistics'):
        """
        Saves the statistics of the simulation to a file named 'statistics.txt'.
        """
        self.stats["makespan"] = self.current_time
        # Assuming throughput and waiting/turnaround times are calculated during the run
        with open(f"{file_name}.txt", "w") as file:
            file.write(f"Makespan: {self.stats['makespan']}\n")
            # file.write(f"Processor Utilization: {self.stats['processor_utilization']}\n")
            # file.write(f"Throughput: {self.stats['throughput']}\n")
            # file.write(f"Average Waiting Time: {self.stats['avg_waiting_time']}\n")
            # file.write(f"Average Turnaround Time: {self.stats['avg_turnaround_time']}\n")


# # Example usage:
# tasks = [
#     ("A", 3, []),
#     ("B", 2, ["A"]),
#     ("C", 1, ["A"]),
#     ("D", 2, ["B", "C"])
# ]
# num_processors = 2
# # algorithm = SimpleQueueAlgorithm()
# algorithm=MinimalRuntimeAlgorithm()
# # algorithm=MaxOutdegreeAlgorithm()
# simulator = SchedulerSimulator(tasks, num_processors, algorithm)
# simulator.run()
# simulator.save_statistics()