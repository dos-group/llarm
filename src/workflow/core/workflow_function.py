from inspect import getfullargspec, iscoroutinefunction

class WorkflowFunction:
    def __init__(self, callable, name= None, hidden=True, argument_types=None, return_type=None):
        self.__callable_specification = getfullargspec(callable)
        self.__callable = callable
        self.__name = name or callable.__name__
        self.__hidden = hidden

        self.__argument_types = argument_types
        if argument_types is None:
            context_filtered_argument_keys = [*self.__callable_specification.args]

            if self.has_self():
                del context_filtered_argument_keys[0]

            if self.has_context():
                del context_filtered_argument_keys[context_index]

            self.__argument_types = {
                argument_key:self.__callable_specification.annotations[argument_key] for argument_key in context_filtered_argument_keys
            }

        self.__return_type = return_type
        if "return" in self.__callable_specification.annotations:
            self.__return_type = self.__callable_specification.annotations["return"]

    @property
    def name(self):
        return self.__name

    @property
    def hidden(self):
        return self.__hidden

    @property
    def callable(self):
        return self.__callable

    @property
    def argument_types(self):
        return self.__argument_types

    def has_self(self):
        return len(self.__callable_specification.args) > 0 and self.__callable_specification.args[0] == "self"

    def context_index(self):
        if self.has_self():
            return 1

        return 0

    def has_context(self):
        return len(self.__callable_specification.args) > self.context_index() and self.__callable_specification.args[self.context_index() ] == "context"

    def is_asynchronous(self):
        return iscoroutinefunction(self.__callable)

    @property
    def return_type(self):
        return self.__return_type

    def stub(name, return_value = None):
        def wrapper(*args):
            print('Execute "{}" and arguments {}'.format(name, ', '.join(map(lambda argument: '"' + str(argument) + '"', args))))

            return return_value

        return wrapper
