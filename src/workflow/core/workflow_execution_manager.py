class WorkflowExecutionManager:
    """
    Manages the execution of a single task and provides mechanisms to interrupt or stop it when required.
    """
    def __init__(self, task):
        self.__task = task

    def stop():
        self.__task.cancel()
