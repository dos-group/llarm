from . import WorkflowFunction

class WorkflowFunctions:
    """
    Aggregates and manages the set of functions accessible to the `WorkflowManager`
    during workflow execution.
    """
    def __init__(self):
        self.__functions = {}

    @property
    def functions(self):
        return self.__functions.values()

    def register(self, *args, **kwargs):
        functions = list(map(lambda callable: WorkflowFunction(callable, **kwargs), args))

        for function in functions:
            if function.name in self.__functions:
                raise Exception("Function name is already in use")

        for function in functions:
            self.__functions[function.name] = function
