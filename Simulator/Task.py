class Task:
    def __init__(self, task_id, runtime, in_degree,out_neighbors):
        self.task_id = task_id
        self.runtime = runtime
        self.in_degree=in_degree
        self.out_neighbors = out_neighbors
        self.out_degree=len(out_neighbors)
        self.end_time = None
        self.processor = None

    def __repr__(self):
        return f"Task({self.task_id}, {self.runtime}, {self.in_degree},{self.out_neighbors},{self.out_degree}, {self.end_time}, {self.processor})"
