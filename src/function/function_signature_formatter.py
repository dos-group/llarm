from inspect import getfullargspec
from typing import Collection,  get_origin, get_args

####################################################################################################

class FunctionSignatureFormatter:
    def format_type(self, type):
        if type is None:
            return "void"

        if isinstance(type, str):
            return type

        if get_origin(type) is None:
            return str(type)

        if get_origin(type).__name__ == 'Collection':
            types = ", ".join(
                map(
                    lambda forward_argument: forward_argument.__forward_arg__, get_args(type)
                )
            )

            return "Collection<" + types + ">"

        return str(type)

    def format_default(self, specification, argument_type_key):
        if specification.defaults is None:
            return ""

        index = specification.args.index(argument_type_key) - len(specification.defaults)

        if index < 0:
            return ""

        if index >= len(specification.defaults):
            return ""

        return " = " + str(specification.defaults[index])

    def format(self, function):
        specification = getfullargspec(function.callable)

        arguments = ", ".join(
            map(
                lambda argument_type_key: ''.join(
                    [
                        argument_type_key,
                        ": ",
                        self.format_type(function.argument_types[argument_type_key]),
                        self.format_default(specification, argument_type_key)
                    ]
                ),
                function.argument_types,
            )
        )

        return_type = self.format_type(function.return_type)

        return "function " + function.name + "(" + arguments + "): " + return_type
