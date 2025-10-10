class WorkflowExecutionPrinter:
    """
    A utility class that prints the next function scheduled for execution within a workflow.
    """

    def register(self, manager):
        manager.event_listeners.before_execute_function.register(self.__before_execute_function)

    def __before_execute_function(
            self,
            arguments,
            context,
            function,
            keyword_arguments,
            *args,
            **kwargs,
    ):
        print(function.name, arguments, keyword_arguments)
