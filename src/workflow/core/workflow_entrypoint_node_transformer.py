from ast import Attribute, parse, unparse, walk, NodeTransformer, AsyncFunctionDef, copy_location, Await
from textwrap import indent

class WorkflowEntrypointNodeTransformer(NodeTransformer):
    def __init__(self, child):
        self.__child = child

    def visit_Pass(self, node):
        return self.__child
