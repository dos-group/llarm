class State:
    """
    Representation of a state in a state machine.

    A state has a name, associated data, which is important for executing the state inside state machine,
    and callbacks, which are executed after the state has been executed successfully.
    """

    def __init__(
        self,
        name,
        data=None,
        then_callbacks=None,
    ):
        self.__name = name
        self.__data = data or {}
        self.__then_callbacks = then_callbacks or []

    @property
    def name(self):
        return self.__name

    @property
    def data(self):
        return self.__data

    @property
    def then_callbacks(self):
        return self.__then_callbacks

    @then_callbacks.setter
    def then_callbacks(self, then_callbacks):
        self.__then_callbacks = then_callbacks

    def then(self, callback):
        self.__then_callbacks.append(callback)
