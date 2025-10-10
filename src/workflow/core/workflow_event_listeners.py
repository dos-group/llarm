from . import WorkflowEventListener

class WorkflowEventListeners:
    """
    An aggregation class for available events within the workflow subsystem of the `WorkflowManager` class.
    """
    def __init__(self):
        self.__before_execute = WorkflowEventListener()
        self.__after_execute = WorkflowEventListener()
        self.__before_execute_function = WorkflowEventListener()
        self.__after_execute_function = WorkflowEventListener()
        self.__before_execute_source_transformation = WorkflowEventListener()
        self.__after_execute_source_transformation = WorkflowEventListener()
        self.__before_execute_ast_transformation = WorkflowEventListener()
        self.__after_execute_ast_transformation = WorkflowEventListener()

    @property
    def before_execute(self):
        return self.__before_execute

    @property
    def after_execute(self):
        return self.__after_execute

    @property
    def before_execute_function(self):
        return self.__before_execute_function

    @property
    def after_execute_function(self):
        return self.__after_execute_function

    @property
    def before_execute_source_transformation(self):
        return self.__before_execute_source_transformation

    @property
    def after_execute_source_transformation(self):
        return self.__after_execute_source_transformation

    @property
    def before_execute_ast_transformation(self):
        return self.__before_execute_ast_transformation

    @property
    def after_execute_ast_transformation(self):
        return self.__after_execute_ast_transformation

    def create(self, name):
        setattr(self, name, WorkflowEventListener())

    def __getattr__(self, name):
        if not hasattr(self, "__" + name):
            return super().__getattr__(name)

        return getattr(self, "__" + name)

    def __setattr__(self, name, value):
        super().__setattr__("__" + name, value)
