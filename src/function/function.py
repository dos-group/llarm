from inspect import getfullargspec

####################################################################################################

class Function:
    def __init__(self, callable, name = None, argument_types = None, return_type = None):
        self.__callable_specification = getfullargspec(callable)
        self.__callable = callable
        self.__name = name or callable.__name__

        self.__argument_types = argument_types
        if argument_types is None:
            context_filtered_argument_keys = [*self.__callable_specification.args]
            if self.has_context():
                del context_filtered_argument_keys[0]

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
    def callable(self):
        return self.__callable

    @property
    def argument_types(self):
        return self.__argument_types

    def has_context(self):
        return len(self.__callable_specification.args) > 0 and self.__callable_specification.args[0] == "context"

    @property
    def return_type(self):
        return self.__return_type

    def stub(name, return_value = None):
        def wrapper(*args):
            print('Execute "{}" and arguments {}'.format(name, ', '.join(map(lambda argument: '"' + str(argument) + '"', args))))

            return return_value

        return wrapper
