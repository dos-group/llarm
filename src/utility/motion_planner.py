from copy import deepcopy

class MotionPlanner:
    """
    The `MotionPlanner` is used to programmatically create complex sequences of actions.
    """

    def __init__(self, controller, transitions = None):
        if transitions is None:
            transitions = []

        self.__controller = controller
        self.__transitions = transitions

    def __getattr__(self, name):
        if not hasattr(self.__controller, name):
            raise Exception("Unknown method")

        def handle_method(*args, **kwargs):
            self.__transitions.append(
                {
                    "name": name,
                    "arguments": args,
                    "keyword_arguments": kwargs,
                    "then_callbacks": [],
                }
            )

            return self

        return handle_method

    def __deepcopy__(self, memo):
        return MotionPlanner(
            self.__controller,
            deepcopy(self.__transitions)
        )

    def then(self, callback):
        if len(self.__transitions) == 0:
            return self

        self.__transitions[len(self.__transitions) - 1]["then_callbacks"].append(
            callback
        )

        return self

    def apply(self):
        transitions = self.__transitions
        self.__transitions = []

        def handle(controller):
            if len(transitions) == 0:
                return

            transition = transitions.pop(0)

            state = getattr(controller, transition["name"])(
                *transition["arguments"],
                **transition["keyword_arguments"],
            )

            for callback in [handle, *transition["then_callbacks"]]:
                state.then(callback)

        self.__controller.state.then(handle)
