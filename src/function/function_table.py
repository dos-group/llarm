from . import Function
from copy import deepcopy

class FunctionTable:
    def __init__(self, signature_formatter):
        self.__signature_formatter = signature_formatter
        self.__functions = {}

    def register(self, callable, **kwargs):
        function = Function(callable, **kwargs)

        if function.name in self.__functions:
            raise Exception("Function name is already in use")

        self.__functions[function.name] = function

    def format_prompt_specification(self):
        return "\n".join(
            map(
                self.__signature_formatter.format,
                self.__functions.values(),
            )
        )

    def evaluate(self, code, enable_breakpoint = False, context = None, tracing = False):
        if context is None:
            context = {}

        result = {
            "context": context,
        }

        if tracing is True:
            result["trace"] = []

        namespace = {
            function.name:self.__create_callable(function, result, enable_breakpoint) for function in self.__functions.values()
        }

        exec(
            code,
            namespace,
            namespace,
        )

        return result

    def __create_callable(self, function, result, enable_breakpoint):
        def callable(*args, **kwargs):
            trace = {}

            if "trace" in result:
                trace = {
                    "name": function.name,
                    "before_context": deepcopy(result["context"]),
                    "arguments": deepcopy(args),
                    "keyword_arguments": deepcopy(kwargs),
                }

            return_value = None

            if enable_breakpoint is True:
                print(function.name, args, kwargs)
                breakpoint()

            if function.has_context():
                return_value = function.callable(result["context"], *args, **kwargs)
            else:
                return_value = function.callable(*args, **kwargs)

            if "trace" in result:
                trace["return_value"] = deepcopy(return_value)
                trace["after_context"] = deepcopy(result["context"])

                result["trace"].append(trace)

            return return_value

        return callable
