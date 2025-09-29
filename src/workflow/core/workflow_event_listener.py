from inspect import iscoroutinefunction

class WorkflowEventListener:
    def __init__(self, listeners = None):
        if listeners is None:
            listeners = []

        self.__listeners = listeners

    def register(self, listener):
        self.__listeners.append(listener)

    async def trigger(self, *args, **kwargs):
        for listener in self.__listeners:
            if iscoroutinefunction(listener):
                await listener(*args, **kwargs)
            else:
                listener(*args, **kwargs)
