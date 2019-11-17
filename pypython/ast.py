'''
Be aware! Lot's of things are broken! Consider this to be demo-ware.
'''

import typing
from enum import auto, Enum
from typing import Any, List, Optional
from dataclasses import field, dataclass

@dataclass
class AST:
    '''
    lineno: Optional[int]
    col_offset: Optional[int]
    end_lineno: Optional[int]
    end_col_offset: Optional[int]
    '''

    pass

class NodeVisitor:
    '''
    See `ast.NodeVisitor`.
    '''

    def visit(self, node, *args, **kwargs):
        method = f'visit_{node.__class__.__name__}'
        visitor = getattr(self, method, self.generic_visit)

        visitor(node, *args, **kwargs)

    def visit_list(self, nodes, *args, **kwargs):
        for node in nodes:
            self.visit(node, *args, **kwargs)

    def visit_tuple(self, nodes, *args, **kwargs):
        for node in nodes:
            self.visit(node, *args, **kwargs)

    def generic_visit(self, node, *args, **kwargs):
        if isinstance(node, AST):
            for field in node.__dataclass_fields__.values():
                self.visit(field.value, *args, **kwargs)

class NodeTransformer:
    '''
    See `ast.NodeTransformer`.
    '''

    def visit(self, node: AST, *args, **kwargs):
        method = f'visit_{node.__class__.__name__}'
        visitor = getattr(self, method, self.generic_visit)

        return visitor(node, *args, **kwargs)

    def visit_list(self, nodes, *args, **kwargs):
        return [ self.visit(node, *args, **kwargs) for node in nodes ]

    def visit_tuple(self, nodes, *args, **kwargs):
        return tuple( self.visit(node, *args, **kwargs) for node in nodes )

    def generic_visit(self, node, *args, **kwargs):
        if isinstance(node, AST):
            fields = {}

            for field in node.__dataclass_fields__:
                value = self.visit(field.value, *args, **kwargs)
                fields[field.name] = value

            return node.__class__(**fields)

        return node

'''
-- ASDL's 5 builtin types are:
-- identifier, int, string, constant
'''

@dataclass
class Ident(AST):
    value: str

    def __repr__(self):
        return f'{self.value}'

@dataclass
class String(AST):
    value: str

'''
module Python
{
    mod = Module(stmt* body, type_ignore *type_ignores)
        | Interactive(stmt* body)
        | Expression(expr body)
        | FunctionType(expr* argtypes, expr returns)
'''

@dataclass
class Stmt(AST):
    pass

@dataclass
class TypeIgnore(AST):
    '''
        type_ignore = TypeIgnore(int lineno, string tag)
    '''

    lineno: int
    tag: String

@dataclass
class Module(AST):
    body: List[Stmt]
    type_ignores: List[TypeIgnore]

@dataclass
class Interactive(AST):
    body: List[Stmt]

@dataclass
class Global(Stmt):
    names: List[Ident]

@dataclass
class Nonlocal(Stmt):
    names: List[Ident]

@dataclass
class Pass(Stmt):
    pass

@dataclass
class Break(Stmt):
    pass

@dataclass
class Continue(Stmt):
    pass

@dataclass
class Expr(AST):
    pass

@dataclass
class Expression(AST):
    body: Expr

@dataclass
class NamedExpr(Expr):
    '''
             | NamedExpr(expr target, expr value)
    '''

    target: Expr
    value: Expr

class Operator(Enum):
    '''
        operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
                     | RShift | BitOr | BitXor | BitAnd | FloorDiv
    '''

    Add = auto()
    Sub = auto()
    Mult = auto()
    MatMult = auto()
    Div = auto()
    Mod = auto()
    Pow = auto()
    LShift = auto()
    RShift = auto()
    BitOr = auto()
    BitXor = auto()
    BitAnd = auto()
    FloorDiv = auto()

    def __repr__(self):
        return f'{self.name}()'

@dataclass
class BinOp(Expr):
    left: Expr
    op: Operator
    right: Expr

class UnaryOper(Enum):
    '''
        unaryop = Invert | Not | UAdd | USub
    '''

    Invert = auto()
    Not = auto()
    UAdd = auto()
    USub = auto()

    def __repr__(self):
        return f'{self.name}()'

@dataclass
class UnaryOp(Expr):
    '''
             | UnaryOp(unaryop op, expr operand)
    '''
    op: UnaryOper
    operand: Expr

@dataclass
class arg(AST):
    '''
        arg = (identifier arg, expr? annotation, string? type_comment)
               attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    '''

    arg: Ident
    annotation: Optional[Expr] = None
    type_comment: Optional[String] = None

@dataclass
class arguments(AST):
    '''
        arguments = (Arg* posonlyargs, Arg* args, Arg? vararg, Arg* kwonlyargs,
                     Expr* kw_defaults, Arg? kwarg, Expr* defaults)
    '''

    posonlyargs: List[arg]
    args: List[arg]
    varargs: Optional[arg]
    kwonlyargs: List[arg]
    kw_defaults: List[Expr]
    kwarg: Optional[arg]
    defaults: List[Expr]

@dataclass
class Lambda(Expr):
    '''
             | Lambda(arguments args, Expr body)
    '''

    args: arguments
    body: Expr

@dataclass
class IfExpr(Expr):
    '''
             | IfExpr(Expr test, Expr body, Expr orelse)
    '''

    test: Expr
    body: Expr
    orelse: Expr

@dataclass
class Dict(Expr):
    '''
             | Dict(expr* keys, expr* values)
    '''

    keys: List[Expr]
    values: List[Expr]

@dataclass
class Set(Expr):
    '''
             | Set(Expr* elts)
    '''

    elts: List[Expr]

@dataclass
class comprehension(AST):
    '''
        comprehension = (Expr target, Expr iter, Expr* ifs, int is_async)
    '''

    target: Expr
    iter: Expr
    ifs: List[Expr]
    is_async: bool = False

@dataclass
class ListComp(Expr):
    '''
             | ListComp(Expr elt, comprehension* generators)
    '''

    elt: Expr
    generators: List[comprehension]

@dataclass
class SetComp(Expr):
    '''
             | SetComp(Expr elt, comprehension* generators)
    '''

    elt: Expr
    generators: List[comprehension]

@dataclass
class DictComp(Expr):
    '''
             | DictComp(expr key, expr value, comprehension* generators)
    '''

    key: Expr
    value: Expr
    generators: List[comprehension]

@dataclass
class GeneratorExp(Expr):
    '''
             | GeneratorExp(Expr elt, comprehension* generators)
    '''

    elt: Expr
    generators: List[comprehension]

@dataclass
class Await(Expr):
    '''
             -- the grammar constrains where yield expressions can occur
             | Await(expr value)
    '''

    value: Expr

@dataclass
class Yield(Expr):
    '''
             | Yield(expr? value)
    '''

    value: Optional[Expr] = None

@dataclass
class YieldFrom(Expr):
    '''
             | YieldFrom(expr value)
    '''

    value: Expr

class CmpOp(Enum):
    '''
        cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
    '''

    Eq = auto()
    NotEq = auto()
    Lt = auto()
    Lte = auto()
    Gt = auto()
    Gte = auto()
    Is = auto()
    IsNot = auto()
    In = auto()
    NotIn = auto()

    def __repr__(self):
        return f'{self.name}()'

@dataclass
class Compare(Expr):
    left: Expr
    ops: List[CmpOp]
    comparators: List[Expr]

@dataclass
class keyword(AST):
    arg: Optional[Ident]
    value: Expr

@dataclass
class Call(Expr):
    func: Expr
    args: List[Expr]
    keywords: List[keyword]


@dataclass
class FormattedValue(Expr):
    '''
             | FormattedValue(expr value, int? conversion, expr? format_spec)
    '''

    value: Expr
    conversion: Optional[int] = None
    format_spec: Optional[Expr] = None

@dataclass
class JoinedStr(Expr):
    '''
             | JoinedStr(expr* values)
    '''

    values: List[Expr]

@dataclass
class Constant(Expr):
    '''
             | Constant(constant value, string? kind)
    '''

    value: Any
    kind: Optional[String] = None

class ExprContext(Enum):
    '''
        ExprContext = Load | Store | Del | AugLoad | AugStore | Param
    '''

    Load = auto()
    Store = auto()
    Del = auto()
    AugLoad = auto()
    AugStore = auto()
    Param = auto()

    def __repr__(self):
        return f'{self.name}()'

@dataclass
class Attribute(Expr):
    '''
             -- the following expression can appear in assignment context
             | Attribute(expr value, identifier attr, ExprContext ctx)
    '''

    value: Expr
    attr: Ident
    #ctx: Optional[ExprContext] = None

@dataclass
class Slice(AST):
    pass

@dataclass
class NormalSlice(Slice):
    '''
        slice = Slice(expr? lower, expr? upper, expr? step)
    '''

    lower: Optional[Expr]
    upper: Optional[Expr]
    step: Optional[Expr]

@dataclass
class ExtSlice(Slice):
    '''
              | ExtSlice(slice* dims)
    '''
    dims: List[Slice]

@dataclass
class Index(Slice):
    '''
              | Index(expr value)
    '''

    value: Expr

@dataclass
class Subscript(Expr):
    '''
             | Subscript(expr value, slice slice, ExprContext ctx)
    '''

    value: Expr
    slice: Slice
    #ctx: Optional[ExprContext] = None

@dataclass
class Starred(Expr):
    '''
             | Starred(expr value, ExprContext ctx)
    '''

    value: Expr
    #ctx: Optional[ExprContext] = None

@dataclass
class Name(Expr):
    '''
             | Name(identifier id, ExprContext ctx)
    '''

    id: Ident
    #ctx: Optional[ExprContext] = None

@dataclass
class ListExpr(Expr):
    '''
             | List(Expr* elts, ExprContext ctx)
    '''

    elts: List[Expr]
    #ctx: Optional[ExprContext] = None

    def __repr__(self):
        return f'List(elts={self.elts!r})'

@dataclass
class Tuple(Expr):
    '''
             | Tuple(Expr* elts, ExprContext ctx)
    '''

    elts: List[Expr]
    #ctx: Optional[ExprContext] = None

class BoolOper(Enum):
    '''
        boolop = And | Or
    '''

    And = auto()
    Or = auto()

    def __repr__(self):
        return f'{self.name}()'

@dataclass
class BoolOp(Expr):
    '''
              -- BoolOp() can use left & right?
        expr = BoolOp(Boolop op, Expr* values)
    '''

    op: BoolOper
    values: List[Expr]

@dataclass
class ExceptHandler(AST):
    '''
        excepthandler = ExceptHandler(expr? type, identifier? name, stmt* body)
                        attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    '''

    typ: Optional[Expr]
    name: Optional[Ident]
    body: List[Stmt]

@dataclass
class alias(AST):
    '''
        -- import name with optional 'as' alias.
        alias = (identifier name, identifier? asname)
    '''

    name: Ident
    asname: Optional[Ident] = None

@dataclass
class FunctionDef(Stmt):
    '''
    stmt = FunctionDef(identifier name, arguments args,
                       stmt* body, expr* decorator_list, expr? returns,
                       string? type_comment)

    stmt = FunctionDef(Ident name, Arguments args,
                       Stmt* body, Expr* decorator_list, Expr? returns,
                       String? type_comment)
    '''

    name: Ident
    args: arguments
    body: List[Stmt]
    decorators: List[Expr]
    returns: Optional[Expr] = None
    type_comment: Optional[String] = None

@dataclass
class AsyncFunctionDef(Stmt):
    '''
          | AsyncFunctionDef(Ident name, Arguments args,
                             Stmt* body, Expr* decorator_list, Expr? returns,
                             String? type_comment)
    '''

    name: Ident
    args: arguments
    body: List[Stmt]
    decorators: List[Expr]
    returns: Optional[Expr] = None
    type_comment: Optional[String] = None

@dataclass
class ClassDef(Stmt):
    '''
          | ClassDef(Ident name,
             Expr* bases,
             reyword* keywords,
             Stmt* body,
             Expr* decorators)
    '''

    name: Ident
    bases: List[Expr]
    keywords: List[keyword]
    body: List[Stmt]
    decorators: List[Expr]

@dataclass
class Return(Stmt):
    '''
          | Return(Expr? value)
    '''

    value: Optional[Expr] = None

@dataclass
class Delete(Stmt):
    targets: List[Expr]

@dataclass
class Assign(Stmt):
    '''
              | Assign(Expr* targets, Expr value, String? type_comment)
    '''

    targets: List[Expr]
    value: Expr
    type_comment: Optional[String] = None

@dataclass
class AugAssign(Stmt):
    '''
              | AugAssign(Expr target, Operator op, Expr value)
    '''

    target: Expr
    op: Operator
    value: Expr

@dataclass
class AnnAssign(Stmt):
    '''
              -- 'simple' indicates that we annotate simple name without parens
              | AnnAssign(Expr target, Expr annotation, Expr? value, int simple)
    '''

    target: Expr
    annotation: Expr
    value: Optional[Expr] = None
    simple: bool = False





###############################################################3


# DEMO

@dataclass
class For(Stmt):
    target: Expr
    iter: Expr
    body: List[Stmt]
    cond: Optional[Expr]
    orelse: List[Stmt]
    type_comment: Optional[String] = None


###############################################################3






@dataclass
class AsyncFor(Stmt):
    '''
              | AsyncFor(Expr target, Expr iter, Stmt* body, Stmt* orelse, String? type_comment)
    '''

    target: Expr
    iter: Expr
    body: List[Stmt]
    orelse: List[Stmt]
    type_comment: Optional[String] = None

@dataclass
class While(Stmt):
    '''
              | While(Expr test, Stmt* body, Stmt* orelse)
    '''

    test: Expr
    body: List[Stmt]
    orelse: List[Stmt]

@dataclass
class If(Stmt):
    '''
              | If(Expr test, Stmt* body, Stmt* orelse)
    '''

    test: Expr
    body: List[Stmt]
    orelse: List[Stmt]

@dataclass
class withitem(AST):
    context_expr: Expr
    optional_vars: Optional[Expr] = None

@dataclass
class With(Stmt):
    items: List[withitem]
    body: List[Stmt]
    type_comment: Optional[String] = None

@dataclass
class AsyncWith(Stmt):
    items: List[withitem]
    body: List[Stmt]
    type_comment: Optional[String] = None

@dataclass
class Raise(Stmt):
    exc: Optional[Expr] = None
    cause: Optional[Expr] = None

@dataclass
class Try(Stmt):
    body: List[Stmt]
    handlers: List[ExceptHandler]
    orelse: List[Stmt]
    finalbody: List[Stmt]

@dataclass
class Assert(Stmt):
    test: Expr
    msg: Optional[Expr] = None

@dataclass
class Import(Stmt):
    names: List[alias]

@dataclass
class ImportFrom(Stmt):
    module: Optional[Ident]
    names: List[alias]
    level: Optional[int]

@dataclass
class ExprStmt(Stmt):
    '''
              | Expr(Expr value)
    '''

    value: Expr

    def __repr__(self):
        return f'Expr(value={self.value!r})'
