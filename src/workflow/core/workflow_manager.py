from . import WorkflowEvent, WorkflowFunctions, WorkflowEventListeners, WorkflowEntrypointNodeTransformer, WorkflowAsynchronousNodeTransformer, WorkflowExecutionManager
from copy import deepcopy
from ast import parse, unparse
from textwrap import indent
from asyncio import get_running_loop, current_task, sleep, timeout

class WorkflowManager:
    def __init__(self):
        self.__functions = WorkflowFunctions()
        self.__event_listeners = WorkflowEventListeners()

    @property
    def functions(self):
        return self.__functions

    @property
    def event_listeners(self):
        return self.__event_listeners

    async def execute(self, source, timeout_in_seconds=None, context=None):
        if context is None:
            context = {}

        transformed_source = await self.__transform_source(source)

        result = await self.event_listeners.before_execute.trigger(source=transformed_source)
        if result is not None:
            transformed_source = result

        execution_manager = WorkflowExecutionManager(current_task())

        namespace = {
            function.name:self.__create_callable(function, execution_manager, context) for function in self.__functions.functions
        }

        exec(
            transformed_source,
            namespace,
            namespace,
        )

        try:
            async with timeout(timeout_in_seconds):
                await namespace["__workflow__"]()
        finally:
            await self.event_listeners.after_execute.trigger()

        return context

    def __create_callable(self, function, execution_manager, context):
        async def callable(*args, **kwargs):
            event = WorkflowEvent(
                call=True,
            )

            await self.event_listeners.before_execute_function.trigger(
                event=event,
                execution_manager=execution_manager,
                function=function,
                arguments=args,
                keyword_arguments=kwargs,
                context=context,
            )

            if event.call is True:
                return_value = None

                if function.is_asynchronous():
                    if function.has_context():
                        return_value = await function.callable(context, *args, **kwargs)
                    else:
                        return_value = await function.callable(*args, **kwargs)
                else:
                    if function.has_context():
                        return_value = function.callable(context, *args, **kwargs)
                    else:
                        return_value = function.callable(*args, **kwargs)

            await self.event_listeners.after_execute_function.trigger(
                event=event,
                execution_manager=execution_manager,
                function=function,
                arguments=args,
                keyword_arguments=kwargs,
                return_value=return_value,
                context=context,
            )

            return return_value

        return callable

    async def __transform_source(self, source):
        await self.event_listeners.before_execute_source_transformation.trigger(
            source=source,
        )

        ast = WorkflowEntrypointNodeTransformer(
            await self.__transform_ast(parse(source))
        ).visit(
            parse("async def __workflow__():\n  pass"),
        )

        source = unparse(ast)

        await self.event_listeners.after_execute_source_transformation.trigger(
            source=source,
        )

        return source

    async def __transform_ast(self, ast):
        await self.event_listeners.before_execute_ast_transformation.trigger(
            ast=ast,
        )

        asynchronous_function_names = set()

        for function in self.__functions.functions:
            asynchronous_function_names.add(function.name)

        ast = WorkflowAsynchronousNodeTransformer(asynchronous_function_names).visit(ast)

        await self.event_listeners.after_execute_ast_transformation.trigger(
            ast=ast,
        )

        return ast
