import inspect
import os
from pycparser import parse_file


def parse_c_file(filename):
    """Parse a c file to a pycparser ast.
    
    Also deleting all unecessary typedefs.
    :param filename: file to parse 
    :return: ast
    """
    # path to fake c libs
    parent = inspect.getfile(inspect.currentframe()).split("/")[-1][:-3]
    cpp_args_path = r"-I{}/utils/fake_libc_include".format(parent)
    # the parsing itself
    ast = parse_file(filename, use_cpp=True, cpp_path="gcc", cpp_args=["-E", cpp_args_path])
    # deleting unecessary childrens
    ast.ext = list(filter(lambda e: e.coord.file == filename, ast.ext))
    return ast
