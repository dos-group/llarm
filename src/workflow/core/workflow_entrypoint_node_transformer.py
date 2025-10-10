from ast import Attribute, parse, unparse, walk, NodeTransformer, AsyncFunctionDef, copy_location, Await
from textwrap import indent

class WorkflowEntrypointNodeTransformer(NodeTransformer):
    """
    A `NodeTransformer` that wraps code inside a specified child node to define
    a clear and structured entry point for workflow execution.
    """
    def __init__(self, child):
        self.__child = child

    def visit_Pass(self, node):
        return self.__child
