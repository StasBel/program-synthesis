from pycparser import c_ast


class Node:
    __slots__ = ["vec", "childs"]

    def __init__(self, vec, childs=None):
        self.vec = vec
        self.childs = childs or list()


class TreeBuilder(c_ast.NodeVisitor):
    def __init__(self, embedder):
        self.embedder = embedder

    def visit(self, node):
        vec = self.embedder.vec(node.__class__.__name__)
        childs = list(self.visit(child) for _, child in node.children())
        return Node(vec, childs)


def make_tree(ast, embedder):
    return TreeBuilder(embedder).visit(ast)
