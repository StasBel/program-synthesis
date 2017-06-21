from pycparser import c_ast

from lstm.tree_rnn import Node


class LabelContex:
    def __init__(self):
        self.label_dict, self.k = dict(), 0

    def process_node(self, node):
        label = node.__class__.__name__
        if label not in self.label_dict:
            self.label_dict[label] = self.k
            self.k += 1
        return self.label_dict[label]


class TreeBuilder(c_ast.NodeVisitor):
    def __init__(self, ctx):
        self.ctx = ctx

    def visit(self, node):
        my_node = Node(self.ctx.process_node(node))
        children, degree = [], 0
        for _, child in node.children():
            child_node, child_degree = self.visit(child)
            children.append(child_node)
            degree = max(degree, child_degree)
        if children:
            my_node.add_children(children)
        return my_node, degree


def translate_ast(ast, ctx):
    tb = TreeBuilder(ctx)
    tree, degree = tb.visit(ast)
    return tree, degree
