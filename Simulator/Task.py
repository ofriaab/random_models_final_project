class Task:
    def __init__(self, task_id, runtime, dependencies):
        self.task_id = task_id
        self.runtime = runtime
        self.dependencies = dependencies
        self.end_time = None
        self.processor = None

    def __repr__(self):
        return f"Task({self.task_id}, {self.runtime}, {self.dependencies}, {self.end_time}, {self.processor})"
