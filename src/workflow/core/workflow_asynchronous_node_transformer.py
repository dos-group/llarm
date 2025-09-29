from ast import Attribute, parse, unparse, walk, NodeTransformer, AsyncFunctionDef, copy_location, Await
from textwrap import indent
from inspect import iscoroutinefunction

class WorkflowAsynchronousNodeTransformer(NodeTransformer):
    def __init__(self, asynchronous_function_names = None):
        if asynchronous_function_names is None:
            asynchronous_function_names = set()

        self.__asynchronous_function_names = asynchronous_function_names

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)

        self.__asynchronous_function_names.add(node.name)

        return copy_location(
            AsyncFunctionDef(
                name=node.name,
                args=node.args,
                body=node.body,
                returns=node.returns,
                decorator_list=node.decorator_list,
                type_comment=node.type_comment,
            ),
            node,
        )

    def visit_Call(self, node):
        node = self.generic_visit(node)

        if isinstance(node.func, Attribute):
            return node

        if node.func.id not in self.__asynchronous_function_names:
            return node

        return copy_location(
            Await(node),
            node,
        )
