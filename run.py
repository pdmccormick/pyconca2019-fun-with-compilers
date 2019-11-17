#!/usr/bin/env python

import sys
import argparse
import traceback

import astor

from pypython.driver import (
        parse_and_compile,
        parse_source,
        to_c_ast,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--source', action='store_true', help='dump processed source')
    parser.add_argument('-PT', '--python-tree', action='store_true', help='dump CPython Abstract Syntax Tree')
    parser.add_argument('-CT', '--c-tree', action='store_true', help='dump CPython Abstract Syntax Tree')
    parser.add_argument('filename', type=str, help='source file')
    opts = parser.parse_args()

    with open(opts.filename, 'r') as fobj:
        source = fobj.read()

    try:
        if opts.source:
            tree = parse_source(source)
            c_tree = to_c_ast(tree)
            print(astor.to_source(c_tree))
        elif opts.python_tree:
            tree = parse_source(source)
            print(tree)
        elif opts.c_tree:
            tree = parse_source(source)
            c_tree = to_c_ast(tree)
            print(astor.dump_tree(c_tree))
        else:
            code = parse_and_compile(source, opts.filename)
            exec(code)
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        print()

        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback)
        print()

        import pdb; pdb.post_mortem()
