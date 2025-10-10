class WorkflowEvent:
    """
    A class representing a single event that encapsulates event-related data within the workflow subsystem.
    """
    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
