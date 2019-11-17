import ast as c_ast

import astor

from .lex import CPythonLexer as Lexer
from .parser import PythonParser
from .cpython_bridge import CPythonAstTransformer

def parse_source(source):
    lexer = Lexer.from_string(source)
    parser = PythonParser(lexer)
    return parser.parse_file_input()

def to_c_ast(node):
    xform = CPythonAstTransformer()
    return xform.transform(node)

def dump_tree(node):
    c_node = to_c_ast(node)
    return astor.dump_tree(c_node)

def to_source(node):
    c_node = to_c_ast(node)
    return astor.to_source(c_node)

def parse_and_compile(source, filename, mode='exec'):
    tree = parse_source(source)
    target_source = to_source(tree)

    return compile(target_source, filename, mode)
