from copy import deepcopy

class WorkflowExecutionTracer:
    def __init__(self):
        self.__traces = []

    @property
    def traces(self):
        return self.__traces

    def register(self, manager):
        manager.event_listeners.before_execute.register(self.__before_execute)
        manager.event_listeners.before_execute_function.register(self.__before_execute_function)
        manager.event_listeners.after_execute_function.register(self.__after_execute_function)

    def __before_execute(self, *args, **kwargs):
        self.__traces = []

    def __before_execute_function(
            self,
            arguments,
            context,
            function,
            keyword_arguments,
            *args,
            **kwargs,
    ):
        trace = {
            "arguments": deepcopy(arguments),
            "before_context": deepcopy(context),
            "function": function,
            "keyword_arguments": deepcopy(keyword_arguments),
        }

        self.__traces.append(trace)

    def __after_execute_function(self, return_value, context, *args, **kwargs):
        last_trace = self.__traces[len(self.__traces) - 1]
        last_trace["return_value"] = deepcopy(return_value)
        last_trace["context"] = deepcopy(context)
