from typing import (
        List,
        Optional,
        Tuple,
        )
from keyword import kwlist as cpython_kwlist
from token import tok_name as cpython_tok_name
from token import *
import ast as C_ast

from . import ast
from .lex import tokens

kwlist = set(cpython_kwlist)

class ExpectedError(SyntaxError):
    def __init__(self, typs, actual):
        super(ExpectedError, self).__init__(f'Got {actual!r} but expected one of {typs!r}')

def parse_number(s):
    return C_ast.literal_eval(s)

    if s.startswith('0x') or s.startswith('0X'):
        return int(s[2:], 16)

    if s.startswith('0o') or s.startswith('0o'):
        return int(s[2:], 8)

    if s.startswith('0b') or s.startswith('0B'):
        return int(s[2:], 2)

    if '.' in s:
        return float(s)

    return int(s)

def parse_string(s):
    return C_ast.literal_eval(s)

    if s.startswith("'''") or s.startswith('"""'):
        return s[3:-3]

    if s.startswith("'''") or s.startswith('"""'):
        return s[3:-3]

class PythonParser:
    def __init__(self, lex):
        self._lex = lex
        self._toks = iter(lex)
        self._next = None

        self.advance()

    def advance(self):
        try:
            self._next = next(self._toks)

            return True
        except StopIteration:
            self._next = None

            return False

    def peek(self, *typs):
        # NB: Allow each element of `typs` to be of type:
        #   - integers (token typs)
        #   - strings (token values)
        #   - sets of integers and strings

        all_typs = []
        for typ in typs:
            if isinstance(typ, ( set, list, tuple )):
                all_typs += list(typ)
            else:
                all_typs.append(typ)

        tok = self._next

        for typ in all_typs:
            if (
                    (isinstance(typ, int) and tok.type == typ and tok.string not in kwlist) or
                    (isinstance(typ, str) and tok.string == typ)
                    ):
                return tok

        return None

    def consume(self, *typs):
        if (tok := self.peek(*typs)):
            self.advance()
            return tok

        return None

    def expect(self, *typs):
        tok = self.consume(*typs)
        if tok is not None:
            return tok

        self.expectedError(typs)

    def expectedError(self, *typs):
        typs = set.union(*[ set(typ) for typ in typs ])
        raise ExpectedError(typs, self._next)

    def parse(self):
        self.expect(ENCODING)
        return expr_stmt(self)

    def parse_single_input(self) -> ast.Interactive:
        self.expect(ENCODING)
        return single_input(self)

    def parse_file_input(self) -> ast.Module:
        self.expect(ENCODING)
        return file_input(self)

    def parse_eval_input(self) -> ast.Expression:
        self.expect(ENCODING)
        return eval_input(self)

def vfpdef(p):
    '''
        vfpdef:
                NAME
            ;
    '''

    return name(p)

Predict_vfpdef = { NAME }

def vfparg(p):
    '''
        vfparg:
            vfpdef [ '=' test ]
            ;
    '''

    defn = vfpdef(p)

    if p.consume('='):
        val = test(p)
    else:
        val = None

    return defn, val

Predict_vfparg = Predict_vfpdef

def varargslist(p):
    '''
        varargslist:
                vfparg
                ( ',' vfparg )*
                [
                    ','
                    [
                            '*' [vfpdef] (',' vfparg )* [',' ['**' vfpdef [',']]]
                        |   '**' vfpdef [',']
                    ]
                ]


            |   '*' [vfpdef] (',' vfparg )* [',' ['**' vfpdef [',']]]
            |   '**' vfpdef [',']
    '''

def vfpdef(p):
    '''
        vfpdef:
                NAME
            ;
    '''

    return name(p)

Predict_vfpdef = { NAME }

def vfparg(p):
    '''
        vfparg:
            vfpdef [ '=' test ]
            ;
    '''

    defn = vfpdef(p)

    if p.consume('='):
        val = test(p)
    else:
        val = None

    return defn, val

Predict_vfparg = Predict_vfpdef

def varargslist_args(p):
    '''
        varargslist_args:
                vfparg
                ( ',' vfparg )*
                [
                    ','
                    [
                            varargslist_starargs
                        |   varargslist_kwargs
                    ]
                ]
            ;
    '''

    arg = vfparg(p)
    args = [ arg ]

    stargs = None
    kwargs = None

    while p.consume(','):
        if p.peek(Predict_vfparg):
            arg = vfparg(p)
            args.append(arg)
            continue

        if p.peek(Predict_varargslist_starargs):
            stargs = varargslist_starargs(p)
        elif p.peek(Predict_varargslist_kwargs):
            kwargs = varargslist_kwargs(p)

        break

    return args, stargs, kwargs

Predict_varargslist_args = Predict_vfparg

def varargslist_starargs(p):
    '''
        varargslist_starargs:
                '*' [ vfpdef ]
                ( ',' vfparg )*
                [
                    ','
                    [ varargslist_kwargs ]
                ]
            ;
    '''

    p.expect('*')

    if p.peek(Predict_vfpdef):
        arg = vfpdef(p)
    else:
        arg = None

    args = [ arg ]
    kwargs = None

    while p.consume(','):
        if p.peek(Predict_vfparg):
            arg = vfparg(p)
            args.append(arg)
            continue

        if p.peek(Predict_varargslist_kwargs):
            kwargs = varargslist_kwargs(p)

        break

    return args, kwargs

Predict_varargslist_starargs = { '*' }

def varargslist_kwargs(p):
    '''
        varargslist_kwargs:
                '**' vfpdef [ ',' ]
            ;
    '''

    p.expect('**')

    defn = vfpdef(p)

    if p.consume(','):
        pass

    return defn

Predict_varargslist_kwargs = { '**' }

def varargslist(p):
    '''
        varargslist:
                varargslist_args
            |   varargslist_starargs
            |   varargslist_kwargs
            ;
    '''

    if p.peek(Predict_varargslist_args):
        return varargslist_args(p)
    elif p.peek(Predict_varargslist_starargs):
        return varargslist_starargs(p)
    elif p.peek(Predict_varargslist_kwargs):
        return varargslist_kwargs(p)
    else:
        return p.expectedError(Predict_varargslist)

Predict_varargslist = Predict_varargslist_args | Predict_varargslist_starargs | Predict_varargslist_kwargs

def atom(p) -> ast.Expr:
    '''
        atom:
                '(' [ yield_expr | testlist_comp ] ')'
            |   '[' [ testlist_comp ] ']'
            |   '{' [ dictorsetmaker ] '}'
            |   '...'
            |   'None'
            |   'True'
            |   'False'
            |   NAME
            |   NUMBER
            |   STRING+
            ;
    '''

    if p.consume('('):
        empty = False
        expr = comp = elts = None

        if p.peek(Predict_yield_expr):
            expr = yield_expr(p)
        elif p.peek(Predict_testlist_comp):
            expr, comp, elts = testlist_comp(p)
        else:
            empty = True

        p.expect(')')

        if empty:
            return ast.Tuple(elts=[])

        if expr is not None:
            return expr

        if comp is not None:
            return ast.GeneratorExp(elt=comp.elt, generators=comp.generators)

        if elts is not None:
            return elts

    elif p.consume('['):
        if p.peek(Predict_testlist_comp):
            expr, comp, elts = testlist_comp(p)
            empty = False
        else:
            empty = True

        p.expect(']')

        if empty:
            return ast.ListExpr(elts=[])

        if expr is not None:
            return ast.ListExpr(elts=[ expr ])

        if comp is not None:
            return comp

        if elts is not None:
            return ast.ListExpr(elts=elts)

    elif p.consume('{'):
        if p.peek(Predict_dictorsetmaker):
            comp = dictorsetmaker(p)
        else:
            comp = None

        p.expect('}')

        return 'dict-or-set-comp', comp

    elif p.consume('...'):
        return ast.Constant(value=Ellipsis)

    elif p.consume('None'):
        return ast.Constant(value=None)

    elif p.consume('True'):
        return ast.Constant(value=True)

    elif p.consume('False'):
        return ast.Constant(value=False)

    elif (tok := p.consume(NAME)):
        return ast.Name(id=tok.string)

    elif (tok := p.consume(NUMBER)):
        return ast.Constant(value=parse_number(tok.string))

    elif (tok := p.consume(STRING)):
        strs = [ parse_string(tok.string) ]
        while (tok := p.consume(STRING)):
            strs.append(parse_string(tok.string))

        return ast.Constant(value=''.join(strs))

Predict_atom = { '(', '[', '{', '...', 'None', 'True', 'False', NAME, NUMBER, STRING }

def atom_expr(p):
    '''
        atom_expr:
                [ 'await' ] atom trailer*
            ;
    '''

    if p.consume('await'):
        is_await = True
    else:
        is_await = False

    arg = atom(p)

    if not p.peek(Predict_trailer):
        return arg

    prev = arg

    while p.peek(Predict_trailer):
        argslist, subs, dotname = trailer(p)

        if argslist is not None:
            args, kwargs = argslist

            prev = ast.Call(func=prev, args=args, keywords=kwargs)

        elif subs is not None:
            prev = ast.Subscript(value=prev, slice=subs)

        elif dotname is not None:
            prev = ast.Attribute(value=prev, attr=dotname)

    if is_await:
        return ast.Await(value=prev)
    else:
        return prev

Predict_atom_expr = { 'await' } | Predict_atom

ArgsList = Tuple[List[ast.Expr], List[ast.keyword]]

# PDM:
#def trailer(p) -> Tuple[Optional[ArgsList], Optional[List[...]], Optional[str]]:
def trailer(p) -> Tuple[Optional[ArgsList], Optional[List], Optional[str]]:
    '''
        trailer:
                '(' [ arglist ] ')'
            |   '[' subscriptlist ']'
            |   '.' NAME
            ;
    '''

    if p.consume('('):
        if p.peek(Predict_arglist):
            args = arglist(p)
        else:
            args = ( [], [] )

        p.expect(')')

        return args, None, None

    elif p.consume('['):
        subs = subscriptlist(p)
        p.expect(']')

        return None, subs, None

    elif p.consume('.'):
        n = name(p)
        return None, None, n

    else:
        return p.expectedError(Predict_trailer)

Predict_trailer = { '(', '[', '.' }

def power(p):
    '''
        power:
                atom_expr [ '**' factor ]
            ;
    '''

    left = atom_expr(p)

    if p.consume('**'):
        right = factor(p)

        return ast.BinOp(left=left, op=ast.Operator.Pow, right=right)

    else:
        return left

Predict_power = Predict_atom_expr

def factor(p):
    '''
        factor:
                ( '+' | '-' | '~' ) factor
            |   power
            ;
    '''

    if p.peek('+', '-', '~'):
        if p.consume('+'):
            op = ast.UnaryOper.UAdd
        elif p.consume('-'):
            op = ast.UnaryOper.USub
        elif p.consume('~'):
            op = ast.UnaryOper.Invert

        operand = factor(p)
        return ast.UnaryOp(op=op, operand=operand)

    else:
        return power(p)

Predict_factor = { '+', '-', '~' } | Predict_power

def term(p):
    '''
        term:
                factor ( ( '*' | '@' | '/' | '%' | '//' ) factor )*
            ;
    '''

    left = factor(p)

    while p.peek('*', '@', '/', '%', '//'):
        if p.consume('*'):
            op = ast.Operator.Mult
        elif p.consume('@'):
            op = ast.Operator.MatMult
        elif p.consume('/'):
            op = ast.Operator.Div
        elif p.consume('%'):
            op = ast.Operator.Mod
        elif p.consume('//'):
            op = ast.Operator.FloorDiv

        right = factor(p)

        left = ast.BinOp(left=left, op=op, right=right)

    return left

Predict_term = Predict_factor

def arith_expr(p):
    '''
        arith_expr:
                term ( ( '+' | '-' ) term )*
            ;
    '''

    left = term(p)

    while p.peek('+', '-'):
        if p.consume('+'):
            op = ast.Operator.Add
        elif p.consume('-'):
            op = ast.Operator.Sub

        right = term(p)

        left = ast.BinOp(left=left, op=op, right=right)

    return left

Predict_arith_expr = Predict_term

def shift_expr(p):
    '''
        shift_expr:
                arith_expr ( ( '<<' | '>>' ) arith_expr )*
            ;
    '''

    left = arith_expr(p)

    while p.peek('<<', '>>'):
        if p.consume('<<'):
            op = ast.Operator.LShift
        elif p.consume('>>'):
            op = ast.Operator.RShift

        right = arith_expr(p)

        left = ast.BinOp(left=left, op=op, right=right)

    return left

Predict_shift_expr = Predict_arith_expr

def and_expr(p):
    '''
        and_expr:
                shift_expr ( '&' shift_expr )*
            ;
    '''

    left = shift_expr(p)

    while p.consume('&'):
        right = shift_expr(p)
        left = ast.BinOp(left=left, op=ast.Operator.BitAnd, right=right)

    return left

Predict_and_expr = Predict_shift_expr

def xor_expr(p):
    '''
        xor_expr:
                and_expr ( '^' and_expr )*
            ;
    '''

    left = and_expr(p)

    while p.consume('^'):
        right = and_expr(p)
        left = ast.BinOp(left=left, op=ast.Operator.BitXor, right=right)

    return left

Predict_xor_expr = Predict_and_expr

def expr(p):
    '''
        expr:
                xor_expr ( '|' xor_expr )*
            ;
    '''

    left = xor_expr(p)

    while p.consume('|'):
        right = xor_expr(p)
        left = ast.BinOp(left=left, op=ast.Operator.BitOr, right=right)

    return left

Predict_expr = Predict_xor_expr

def comparison(p):
    '''
        comparison:
                expr ( comp_op expr )*
            ;
    '''

    left = expr(p)

    if not p.peek(Predict_comp_op):
        return left

    ops = []
    comparators = []

    while (peek := p.peek(Predict_comp_op)):
        op = comp_op(p)
        comparator = expr(p)

        ops.append(op)
        comparators.append(comparator)

    return ast.Compare(left=left, ops=ops, comparators=comparators)

Predict_comparison = Predict_expr

def comp_op(p) -> ast.CmpOp:
    '''
        comp_op:
                '<' | '>' | '==' | '>=' | '<=' | '<>' | '!=' | 'in'
            |   'is'
            |   'is' 'not'
            |   'not' 'in'
            ;
    '''

    if p.consume('<'):
        return ast.CmpOp.Lt
    elif p.consume('>'):
        return ast.CmpOp.Gt
    elif p.consume('=='):
        return ast.CmpOp.Eq
    elif p.consume('>='):
        return ast.CmpOp.Gte
    elif p.consume('<='):
        return ast.CmpOp.Lte
    elif p.consume('!='):
        return ast.CmpOp.NotEq
    elif p.consume('in'):
        return ast.CmpOp.In
    elif p.consume('is'):
        if p.consume('not'):
            return ast.CmpOp.IsNot
        else:
            return ast.CmpOp.Is
    elif p.consume('not'):
        return ast.CmpOp.NotIn
    else:
        return p.expectedError(Predict_comp_op)

Predict_comp_op = { '<', '>', '==', '>=', '<=', '<>', '!=', 'in', 'not', 'is' }

def not_test(p):
    '''
        not_test:
                'not' not_test
            |   comparison
            ;
    '''

    if p.consume('not'):
        operand = not_test(p)
        return ast.UnaryOp(op=ast.UnaryOper.Not, operand=operand)
    else:
        return comparison(p)

Predict_not_test =  { 'not' } | Predict_comparison

def and_test(p) -> ast.Expr:
    '''
        and_test:
                not_test ( 'and' not_test )*
            ;
    '''

    value = not_test(p)
    values = [ value ]

    while p.consume('and'):
        value = not_test(p)
        values.append(value)

    if len(values) == 1:
        return values[0]
    else:
        return ast.BoolOp(op=ast.BoolOper.And, values=values)

Predict_and_test = Predict_not_test

def or_test(p) -> ast.Expr:
    '''
        or_test:
                and_test ( 'or' and_test )*
            ;
    '''

    value = and_test(p)
    values = [ value ]

    while p.consume('or'):
        value = and_test(p)
        values.append(value)

    if len(values) == 1:
        return values[0]
    else:
        return ast.BoolOp(op=ast.BoolOper.Or, values=values)

Predict_or_test = Predict_and_test

def lambdef(p):
    '''
        lambdef:
                'lambda' [ varargslist ] ':' test
            ;
    '''

    p.expect('lambda')

    if p.peek(Predict_varargslist):
        args = varargslist(p)
    else:
        args = []

    p.expect(':')

    body = test(p)

    return 'lambda', args, body

Predict_lambdef = { 'lambda' }

def test(p) -> ast.Expr:
    '''
        test:
                or_test [ 'if' or_test 'else' test ]
            |   lambdef
            ;
    '''

    if p.peek(Predict_lambdef):
        return lambdef(p)
    else:
        lhs = or_test(p)

        if p.consume('if'):
            cond = or_test(p)
            p.expect('else')
            rhs = test(p)

            return ast.IfExpr(test=cond, body=lhs, orelse=rhs)

        return lhs

Predict_test = Predict_or_test | Predict_lambdef

def lambdef_nocond(p):
    '''
        lambdef_nocond:
                'lambda' [ varargslist ] ':' test_nocond
            ;
    '''

    p.expect('lambda')

    if p.peek(Predict_varargslist):
        args = varargslist(p)
    else:
        args = []

    p.expect(':')

    body = test_nocond(p)

    return 'lambda', args, t

Predict_lambdef_nocond  = { 'lambda' }

def test_nocond(p):
    '''
        test_nocond:
                or_test
            |   lambdef_nocond
            ;
    '''

    if p.peek(Predict_or_test):
        return or_test(p)
    elif p.peek(Predict_lambdef_nocond):
        return lambdef_nocond(p)
    else:
        return p.expectedError(Predict_test_nocond)

Predict_test_nocond = Predict_or_test | Predict_lambdef_nocond

def sliceop(p) -> Optional[ast.Expr]:
    '''
        sliceop:
                ':' [ test ]
            ;
    '''

    p.expect(':')

    if p.peek(Predict_test):
        return test(p)

    return None

Predict_sliceop = { ':' }

def subscript(p):
    '''
        subscript:
                test
            |   [ test ] ':' [ test ] [ sliceop ]
            ;
    '''

    if p.peek(Predict_test):
        left = test(p)

        if not p.peek(':'):
            return 'subscript', left

    else:
        left = None

    p.consume(':')

    if p.peek(Predict_test):
        right = test(p)
    else:
        right = None

    if p.peek(Predict_sliceop):
        op = sliceop(p)
    else:
        op = None

    return 'slice', left, right, op

Predict_subscript = { ':' } | Predict_test

def subscriptlist(p):
    '''
        subscriptlist:
                subscript ( ',' subscript )* [ ',' ]
            ;
    '''

    sub = subscript(p)
    subs = [ sub ]

    while p.consume(','):
        if not p.peek(Predict_subscript):
            break

        sub = subscript(p)
        subs.append(sub)

    return subs

Predict_subscriptlist = Predict_subscript

def star_expr(p):
    '''
        star_expr:
                '*' expr
            ;
    '''

    p.expect('*')

    expr(p)

Predict_star_expr = { '*' }

def dictorsetmaker(p):
    '''
        dictorsetmaker:
                '**' expr
                    (
                            comp_for
                        |   ( ','
                                (
                                        test ':' test
                                    |   '**' expr
                                )
                            )* [ ',' ]
                    )
            |   test ':' test
                    (
                            comp_for
                        |   ( ','
                                (
                                        test ':' test
                                    |   '**' expr
                                )
                            )* [ ',' ]
                    )
            |   test
                    (
                            comp_for
                        |   ( ','
                                (
                                        test
                                    |   star_expr
                                )
                            )* [ ',' ]
                    )
            |   star_expr
                    (
                            comp_for
                        |   ( ','
                                (
                                        test
                                    |   star_expr
                                )
                            )* [ ',' ]
                    )
            ;
    '''

    if p.consume('**'):
        # dict (first element splat)
        splat = expr(p)

        if p.peek(Predict_comp_for):
            _, comps = comp_for(p)

            return 'dict-comp', ( '**', splat ), comps

        else:
            elts = [
                    ( '**', splat ),
                    ]

            while p.consume(','):
                if p.peek(Predict_test):
                    key = test(p)
                    p.expect(':')
                    val = test(p)

                    elts.append(( 'elem', key, val ))

                elif p.consume('**'):
                    splat = expr(p)

                    elts.append(( '**', splat ))

                else:
                    break

            return 'dict-literal', elts

    elif p.peek(Predict_test):
        # dict or set
        elt = test(p)

        if p.consume(':'):
            # dict
            key = elt
            val = test(p)


            if p.peek(Predict_comp_for):
                _, comps = comp_for(p)

                val = 'comp', val, comp

                elts = [
                        ( 'elt', key, val ),
                        ]

                return 'dict-literal', elts

            elts = [
                    ( 'elt', key, val ),
                    ]

            while p.consume(','):
                if p.peek(Predict_test):
                    key = test(p)
                    p.expect(':')
                    val = test(p)

                    elts.append(( 'elt', key, val ))

                elif p.consume('**'):
                    splat = expr(p)

                    elts.append(( '**', splat ))

                else:
                    break

            return 'dict-literal', elts

        else:
            # set
            '''
            |   test
                    (
                            comp_for
                        |   ( ','
                                (
                                        test
                                    |   star_expr
                                )
                            )* [ ',' ]
            '''

            if p.peek(Predict_comp_for):
                _, comps = comp_for(p)

                return 'set-comp', elt, comp

            elts = [
                    ( 'elt', elt ),
                    ]

            while p.consume(','):
                if p.peek(Predict_test):
                    elt = test(p)
                    elts.append(( 'elt', elt ))

                elif p.peek(Predict_star_expr):
                    e = star_expr(p)
                    elts.append(( '*', e ))

                else:
                    break

            return 'set-literal', elts

    elif p.peek(Predict_star_expr):
        # set (first element splat)
        splat = star_expr(p)

    else:
        return p.expectedError(Predict_dictorsetmaker)

Predict_dictorsetmaker = { '**' } |  Predict_test | Predict_star_expr

def exprlist(p) -> ast.Expr:
    '''
        exprlist:
                ( expr | star_expr ) ( ',' ( expr | star_expr ) )* [ ',' ]
            ;
    '''

    if p.peek(Predict_expr):
        elt = expr(p)
    elif p.peek(Predict_star_expr):
        elt = star_expr(p)
    else:
        return p.expectedError(Predict_expr, Predict_star_expr)

    if not p.peek(','):
        return elt

    elts = [ elt ]

    while p.consume(','):
        if p.peek(Predict_expr):
            elt = expr(p)
        elif p.peek(Predict_star_expr):
            elt = star_expr(p)
        else:
            break

        elts.append(elt)

    return ast.Tuple(elts=elts)

Predict_exprlist = Predict_expr | Predict_star_expr

def namedexpr_test(p):
    '''
        namedexpr_test:
                test [ ':=' test ]
            ;
    '''

    left = test(p)

    if p.consume(':='):
        right = test(p)

        return 'walrus', left, right

    return left

Predict_namedexpr_test = Predict_test

ElemComprehension = Tuple[ast.Expr, List[ast.comprehension]]

def testlist_comp(p) -> Tuple[Optional[ast.Expr], Optional[ast.ListComp], Optional[List[ast.Expr]]]:
    '''
        testlist_comp:
                ( namedexpr_test | star_expr )
                    (       comp_for
                        |   ( ',' ( namedexpr_test | star_expr ) )* [ ',' ]
                    )
            ;
    '''

    comps = []

    # PDM:

    if p.peek(Predict_namedexpr_test):
        elt = namedexpr_test(p)
    elif p.peek(Predict_star_expr):
        elt = star_expr(p)
    else:
        return p.expectedError(Predict_namedexpr_test, Predict_star_expr)

    if p.peek(Predict_comp_for):
        _, comps = comp_for(p)

        return None, ast.ListComp(elt=elt, generators=comps), None

    if not p.peek(','):
        return elt, None, None

    elts = [ elt ]

    while p.consume(','):
        if p.peek(Predict_namedexpr_test):
            elt = namedexpr_test(p)
        elif p.peek(Predict_star_expr):
            elt = star_expr(p)
        else:
            break

        elts.append(elt)

    return None, None, elts

Predict_testlist_comp = Predict_namedexpr_test | Predict_star_expr

def testlist_star_expr(p) -> ast.Expr:
    '''
        testlist_star_expr:
                ( test | star_expr ) ( ',' ( test | star_expr ) )* [ ',' ]
            ;
    '''

    if p.peek(Predict_test):
        elt = test(p)
    elif p.peek(Predict_star_expr):
        elt = star_expr(p)
    else:
        return p.expectedError(Predict_test, Predict_star_expr)

    if not p.peek(','):
        return elt

    elts = [ elt ]

    while p.consume(','):
        if p.peek(Predict_test):
            elt = test(p)
        elif p.peek(Predict_star_expr):
            elt = star_expr(p)
        else:
            break

        elts.append(elt)

    return ast.Tuple(elts=elts)

Predict_testlist_star_expr = Predict_test | Predict_star_expr

def expr_stmt(p) -> ast.Stmt:
    '''
        expr_stmt:
                testlist_star_expr
                    (
                            annassign
                        |   augassign ( yield_expr | testlist )
                        |   [ ( '=' ( yield_expr | testlist_star_expr ) )+ [ TYPE_COMMENT ] ]
                        )
            ;
    '''

    elt = testlist_star_expr(p)

    if p.peek(Predict_annassign):
        return annassign(p, elt)

    elif p.peek(Predict_augassign):
        op = augassign(p)

        if p.peek(Predict_yield_expr):
            value = yield_expr(p)
        elif p.peek(Predict_testlist):
            value = testlist(p)
        else:
            return p.expectedError(Predict_yield_expr, Predict_testlist)

        return ast.AugAssign(target=elt, op=op, value=value)

    else:
        if not p.peek('='):
            return ast.ExprStmt(value=elt)

        elts = [ elt ]
        tc = None

        while p.consume('='):
            if p.peek(Predict_yield_expr):
                elt = yield_expr(p)
            elif p.peek(Predict_testlist_star_expr):
                elt = testlist_star_expr(p)
            else:
                return p.expectedError(Predict_yield_expr, Predict_testlist_star_expr)

            elts.append(elt)

            if (tok := p.consume(TYPE_COMMENT)):
                tc = tok.string
                break

        return ast.Assign(targets=elts[:-1], value=elts[-1], type_comment=tc)

Predict_expr_stmt = Predict_testlist_star_expr

def annassign(p, target: ast.Expr) -> ast.AnnAssign:
    '''
        annassign:
                ':' test [ '=' ( yield_expr | testlist_star_expr ) ]
            ;
    '''

    p.expect(':')

    notate = test(p)

    value = None
    if p.consume('='):
        if p.peek(Predict_yield_expr):
            value = yield_expr(p)
        elif p.peek(Predict_testlist_star_expr):
            value = testlist_star_expr(p)
        else:
            return p.expectedError(Predict_yield_expr, Predict_testlist_star_expr)

    return ast.AnnAssign(target=target, annotation=notate, value=value)

Predict_annassign = ':'

def augassign(p):
    '''
        augassign:
                '+=' | '-=' | '*=' | '@=' | '/=' | '%=' | '&=' | '|=' | '^=' | '<<=' | '>>=' | '**=' | '//='
            ;
    '''

    if p.consume('+='):
        return ast.Operator.Add
    elif p.consume('-='):
        return ast.Operator.Sub
    elif p.consume('*='):
        return ast.Operator.Mult
    elif p.consume('@='):
        return ast.Operator.MatMult
    elif p.consume('/='):
        return ast.Operator.Div
    elif p.consume('%='):
        return ast.Operator.Mod
    elif p.consume('&='):
        return ast.Operator.BitAnd
    elif p.consume('|='):
        return ast.Operator.BitOr
    elif p.consume('^='):
        return ast.Operator.BitXor
    elif p.consume('<<='):
        return ast.Operator.LShift
    elif p.consume('>>='):
        return ast.Operator.RShift
    elif p.consume('**='):
        return ast.Operator.Pow
    elif p.consume('//='):
        return ast.Operator.FloorDiv
    else:
        return p.expectedError(Predict_augassign)

Predict_augassign = { '+=', '-=', '*=', '@=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '**=', '//=' }

def yield_arg(p) -> ast.Expr:
    '''
        yield_arg:
                'from' test
            |   testlist_star_expr
            ;
    '''

    if p.consume('from'):
        value = test(p)
        return ast.YieldFrom(value=value)

    elif p.peek(Predict_testlist_star_expr):
        value = testlist_star_expr(p)
        return ast.Yield(value=value)

    else:
        return p.expectedError(Predict_yield_arg)

Predict_yield_arg = { 'from' } | Predict_testlist_star_expr

def yield_expr(p) -> ast.Expr:
    '''
        yield_expr:
                'yield' [ yield_arg ]
            ;
    '''

    p.expect('yield')

    if p.peek(Predict_yield_arg):
        return yield_arg(p)
    else:
        return ast.Yield()

Predict_yield_expr = { 'yield' }

def del_stmt(p) -> ast.Delete:
    '''
        del_stmt:
                'del' exprlist
            ;
    '''

    p.expect('del')
    targets = exprlist(p)

    return ast.Delete(targets=targets)

Predict_del_stmt = { 'del' }

def pass_stmt(p) -> ast.Pass:
    '''
        pass_stmt:
                'pass'
            ;
    '''

    p.expect('pass')

    return ast.Pass()

Predict_pass_stmt = { 'pass' }

def break_stmt(p) -> ast.Break:
    '''
        break_stmt:
                'break'
            ;
    '''

    p.expect('break')

    return ast.Break()

Predict_break_stmt = { 'break' }

def continue_stmt(p) -> ast.Continue:
    '''
        continue_stmt:
                'continue'
            ;
    '''

    p.expect('continue')

    return ast.Continue()

Predict_continue_stmt = { 'continue' }

def return_stmt(p) -> ast.Return:
    '''
        return_stmt:
                'return' [ testlist_star_expr ]
            ;
    '''

    p.expect('return')

    if p.peek(Predict_testlist_star_expr):
        value = testlist_star_expr(p)
    else:
        value = None

    return ast.Return(value=value)

Predict_return_stmt = { 'return' }

def raise_stmt(p) -> ast.Raise:
    '''
        raise_stmt:
                'raise' [ test [ 'from' test ] ]
            ;
    '''

    p.expect('raise')

    exc = cause = None

    if p.peek(Predict_test):
        exc = test(p)

        if p.consume('from'):
            cause = test(p)

    return ast.Raise(exc=exc, cause=cause)

Predict_raise_stmt = { 'raise' }

def yield_stmt(p) -> ast.Stmt:
    '''
        yield_stmt:
                yield_expr
            ;
    '''

    value = yield_expr(p)

    return ast.ExprStmt(value=value)

Predict_yield_stmt = Predict_yield_expr

def import_name(p) -> ast.Import:
    '''
        import_name:
                'import' dotted_as_names
            ;
    '''

    p.expect('import')

    names = dotted_as_names(p)

    return ast.Import(names=names)

Predict_import_name = { 'import' }

def import_from(p) -> ast.ImportFrom:
    # NB: A `...` will be tokenized as ELLIPSIS, not three separate dots `.`, hence the ( '.' | '...' ) part

    '''
        import_from:
                'from'
                    (
                            dotted_name
                        |   ( '.' | '...' )+ [ dotted_name ]
                    )
                    'import'
                    (
                            '*'
                        |   '(' import_as_names ')'
                        |   import_as_names
                    )
            ;
    '''

    p.expect('from')

    path = []

    # from foo import _
    if p.peek(Predict_dotted_name):
        name = dotted_name(p)
        path = [ name ]

    else:
        # from . import _
        # from .... import _
        while True:
            if p.consume('.'):
                path.append('.')
            elif p.consume('...'):
                path += [ '.', '.', '.' ]
            else:
                return p.expectedError('.', '...')

            if not p.peek('.', '...'):
                break

        # from .foo import _
        # from ....bar import _
        if p.peek(Predict_dotted_name):
            name = dotted_name(p)
            path.append(name)

    p.expect('import')

    star = False
    names = []

    if p.consume('*'):
        star = True

    elif p.consume('('):
        names = import_as_names(p)
        p.expect(')')

    else:
        names = import_as_names(p)

    module = '.'.join(path)

    level = 0  # FIXME?

    return ast.ImportFrom(module=module, names=names, level=level)

Predict_import_from = { 'from' }

def import_stmt(p) -> ast.Stmt:
    '''
        import_stmt:
                import_name
            |   import_from
            ;
    '''

    if p.peek(Predict_import_name):
        return import_name(p)
    elif p.peek(Predict_import_from):
        return import_from(p)
    else:
        return p.expectedError(Predict_import_name, Predict_import_from)

Predict_import_stmt = Predict_import_name | Predict_import_from

def dotted_name(p) -> str:
    '''
        dotted_name:
                NAME ( '.' NAME )*
            ;
    '''

    n = name(p)
    names = [ n ]

    while p.consume('.'):
        n = name(p)
        names.append(n)

    return '.'.join(names)

Predict_dotted_name = { NAME }

def dotted_as_name(p) -> ast.alias:
    '''
        dotted_as_name:
                dotted_name [ 'as' NAME ]
            ;
    '''

    n = dotted_name(p)

    if p.consume('as'):
        asname = name(p)
    else:
        asname = None

    return ast.alias(name=n, asname=asname)

Predict_dotted_as_name = Predict_dotted_name

def dotted_as_names(p) -> List[ast.alias]:
    '''
        dotted_as_names:
                dotted_as_name ( ',' dotted_as_name )*
            ;
    '''

    n = dotted_as_name(p)
    names = [ n ]

    while p.consume(','):
        n = dotted_as_name(p)
        names.append(n)

    return names

Predict_dotted_as_names = Predict_dotted_as_name

def import_as_name(p) -> ast.alias:
    '''
        import_as_name:
                NAME [ 'as' NAME ]
            ;
    '''

    n = name(p)

    if p.consume('as'):
        asname = name(p)
    else:
        asname = None

    return ast.alias(name=n, asname=asname)

Predict_import_as_name = { NAME }

def import_as_names(p) -> List[ast.alias]:
    '''
        import_as_names:
                import_as_name ( ',' import_as_name )* [ ',' ]
            ;
    '''

    n = import_as_name(p)
    names = [ n ]

    while p.consume(','):
        if not p.peek(Predict_import_as_name):
            break

        n = import_as_name(p)
        names.append(n)

    return names

Predict_import_as_names = Predict_import_as_name

def flow_stmt(p):
    '''
        flow_stmt:
                break_stmt
            |   continue_stmt
            |   return_stmt
            |   raise_stmt
            |   yield_stmt
            ;
    '''

    if p.peek(Predict_break_stmt):
        return break_stmt(p)
    elif p.peek(Predict_continue_stmt):
        return continue_stmt(p)
    elif p.peek(Predict_return_stmt):
        return return_stmt(p)
    elif p.peek(Predict_raise_stmt):
        return raise_stmt(p)
    elif p.peek(Predict_yield_stmt):
        return yield_stmt(p)
    else:
        return p.expectedError(Predict_flow_stmt)

Predict_flow_stmt = Predict_break_stmt | Predict_continue_stmt | Predict_return_stmt | Predict_raise_stmt | Predict_yield_stmt

def name(p):
    tok = p.expect(NAME)
    return tok.string

def global_stmt(p) -> ast.Global:
    '''
        global_stmt:
                'global' NAME ( ',' NAME )*
            ;
    '''

    p.expect('global')

    names = []

    n = name(p)
    names.append(n)

    while p.consume(','):
        n = name(p)
        names.append(n)

    return ast.Global(names=names)

Predict_global_stmt = { 'global' }

def nonlocal_stmt(p) -> ast.Nonlocal:
    '''
        nonlocal_stmt:
                'nonlocal' NAME ( ',' NAME )*
            ;
    '''

    p.expect('nonlocal')

    names = []

    n = name(p)
    names.append(n)

    while p.consume(','):
        n = name(p)
        names.append(n)

    return ast.Nonlocal(names=names)

Predict_nonlocal_stmt = { 'nonlocal' }

def assert_stmt(p) -> ast.Assert:
    '''
        assert_stmt:
                'assert' test [ ',' test ]
            ;
    '''

    p.expect('assert')

    cond = test(p)

    if p.consume(','):
        msg = test(p)
    else:
        msg = None

    return ast.Assert(test=cond, msg=msg)

Predict_assert_stmt = { 'assert' }

def small_stmt(p) -> ast.Stmt:
    '''
        small_stmt:
                expr_stmt
            |   del_stmt
            |   pass_stmt
            |   flow_stmt
            |   import_stmt
            |   global_stmt
            |   nonlocal_stmt
            |   assert_stmt
            ;
    '''

    if p.peek(Predict_expr_stmt):
        return expr_stmt(p)
    elif p.peek(Predict_del_stmt):
        return del_stmt(p)
    elif p.peek(Predict_pass_stmt):
        return pass_stmt(p)
    elif p.peek(Predict_flow_stmt):
        return flow_stmt(p)
    elif p.peek(Predict_import_stmt):
        return import_stmt(p)
    elif p.peek(Predict_global_stmt):
        return global_stmt(p)
    elif p.peek(Predict_nonlocal_stmt):
        return nonlocal_stmt(p)
    elif p.peek(Predict_assert_stmt):
        return assert_stmt(p)
    else:
        return p.expectedError(Predict_small_stmt)

Predict_small_stmt = Predict_expr_stmt | Predict_del_stmt | Predict_pass_stmt | Predict_flow_stmt | Predict_import_stmt | Predict_global_stmt | Predict_nonlocal_stmt | Predict_assert_stmt

def simple_stmt(p) -> List[ast.Stmt]:
    '''
        simple_stmt:
                small_stmt ( ';' small_stmt )* [ ';' ] NEWLINE
            ;
    '''

    s = small_stmt(p)
    stmts = [ s ]

    while p.consume(';'):
        if not p.peek(Predict_small_stmt):
            break

        s = small_stmt(p)
        stmts.append(s)

    p.expect(NEWLINE)

    return stmts

Predict_simple_stmt = Predict_small_stmt

def if_stmt(p) -> ast.If:
    '''
        if_stmt:
                'if' namedexpr_test ':' suite ( 'elif' namedexpr_test ':' suite )* [ 'else' ':' suite ]
            ;
    '''

    p.expect('if')
    cond = namedexpr_test(p)
    p.expect(':')
    body = suite(p)

    top = ast.If(test=cond, body=body, orelse=[])
    last = top

    while p.consume('elif'):
        cond = namedexpr_test(p)
        p.expect(':')
        body = suite(p)

        stmt = ast.If(test=cond, body=body, orelse=[])
        last.orelse = [ stmt ]
        last = stmt

    if p.consume('else'):
        p.expect(':')
        last.orelse = suite(p)

    return top

Predict_if_stmt = { 'if' }

def while_stmt(p) -> ast.While:
    '''
        while_stmt:
                'while' namedexpr_test ':' suite [ 'else' ':' suite ]
            ;
    '''

    p.expect('while')

    cond = namedexpr_test(p)
    p.expect(':')

    body = suite(p)

    if p.consume('else'):
        p.expect(':')
        orelse = suite(p)
    else:
        orelse = None

    return ast.While(test=cond, body=body, orelse=orelse)

Predict_while_stmt = { 'while' }





###############################################################3

# DEMO

def for_stmt(p) -> ast.For:
    '''
    for_stmt:
        'for' exprlist 'in' testlist [ 'while' namedexpr_test ] ':' suite
    '''

    p.expect('for')
    target = exprlist(p)
    p.expect('in')
    iters = testlist(p)

    if p.consume('while'):
        cond = namedexpr_test(p)
    else:
        cond = None

    p.expect(':')
    body = suite(p)

    return ast.For(
            target=target,
            iter=iters,
            body=body,
            cond=cond,
            orelse=None,
            type_comment=None)

Predict_for_stmt = { 'for' }

###############################################################3






"""
def for_stmt(p) -> ast.For:
    '''
        for_stmt:
                'for' exprlist 'in' testlist ':' suite
            ;
    '''

    p.expect('for')
    target = exprlist(p)
    p.expect('in')
    iters = testlist(p)
    p.expect(':')
    body = suite(p)

    return ast.For(target=target, iter=iters, body=body, orelse=None, type_comment=None)

Predict_for_stmt = { 'for' }
"""



"""
def for_stmt(p) -> ast.For:
    '''
        for_stmt:
                'for' exprlist 'in' testlist ':' [ TYPE_COMMENT ] suite [ 'else' ':' suite ]
            ;
    '''

    p.expect('for')
    target = exprlist(p)
    p.expect('in')
    iters = testlist(p)
    p.expect(':')

    if (tok := p.consume(TYPE_COMMENT)):
        tc = tok.string
    else:
        tc = None

    body = suite(p)

    if p.consume('else'):
        p.expect(':')
        orelse = suite(p)
    else:
        orelse = None

    return ast.For(target=target, iter=iters, body=body, orelse=orelse, type_comment=tc)

Predict_for_stmt = { 'for' }
"""

def except_clause(p) -> Tuple[Optional[ast.Expr], Optional[str]]:
    # NB compile.c makes sure that the default except clause is last
    '''
        except_clause:
                'except' [ test [ 'as' NAME ] ]
            ;
    '''

    p.expect('except')

    typ = asname = None

    if p.peek(Predict_test):
        typ = test(p)

        if p.consume('as'):
            asname = name(p)

    return typ, asname

Predict_except_clause = { 'except' }

def try_stmt(p) -> ast.Try:
    '''
        try_stmt:
                'try' ':' suite
                (
                    ( except_clause ':' suite )+
                    [ 'else' ':' suite ]
                    [ 'finally' ':' suite ]
                |   'finally' ':' suite
                )
            ;
    '''

    p.expect('try')
    p.expect(':')
    body = suite(p)

    finalbody = None
    orelse = None
    handlers = []

    if p.consume('finally'):
        p.expect(':')
        finalbody = suite(p)
    else:
        while True:
            typ, asname = except_clause(p)
            p.expect(':')
            body = suite(p)

            handler = ast.ExceptHandler(typ=typ, name=asname, body=body)
            handlers.append(handler)

            if not p.peek(Predict_except_clause):
                break

        if p.consume('else'):
            p.expect(':')
            orelse = suite(p)

        if p.consume('finally'):
            p.expect(':')
            finalbody = suite(p)

    return ast.Try(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody)

Predict_try_stmt = { 'try' }

def with_item(p) -> ast.withitem:
    '''
        with_item:
                test [ 'as' expr ]
            ;
    '''

    e = test(p)

    if p.consume('as'):
        var = expr(p)
    else:
        var = None

    return ast.withitem(context_expr=e, optional_vars=var)

Predict_with_item = Predict_test

def with_stmt(p) -> ast.With:
    '''
        with_stmt:
                'with' with_item ( ',' with_item )*  ':' [ TYPE_COMMENT ] suite
            ;
    '''

    p.expect('with')

    item = with_item(p)
    items = [ item ]

    while p.consume(','):
        item = with_item(p)
        items.append(item)

    p.expect(':')

    if (tok := p.peek(TYPE_COMMENT)):
        tc = tok.string
    else:
        tc = None

    body = suite(p)

    return ast.With(items=items, body=body, type_comment=tc)

Predict_with_stmt = { 'with' }

def tfpdef(p):
    '''
        tfpdef:
                NAME [ ':' test ]
            ;
    '''

    n = name(p)

    if p.consume(':'):
        typ = test(p)
    else:
        typ = None

    return n, typ

Predict_tfpdef = { NAME }

def tfparg(p):
    '''
        tfparg:
            tfpdef [ '=' test ]
            ;
    '''

    defn = tfpdef(p)

    if p.consume('='):
        val = test(p)
    else:
        val = None

    return defn, val

Predict_tfparg = Predict_tfpdef

def typedargslist_args(p):
    '''
        typedargslist_args:
                tfparg
                ( ',' [ TYPE_COMMENT ] tfparg )*
                (
                        TYPE_COMMENT
                    |   [
                            ',' [ TYPE_COMMENT ]
                            [
                                    typedargslist_starargs
                                |   typedargslist_kwargs
                            ]
                        ]
                )
            ;
    '''

    arg = tfparg(p)
    args = [ arg ]

    typs = []

    end = False

    stargs = None
    kwargs = None

    while p.consume(','):
        if (tok := p.peek(TYPE_COMMENT)):
            typ = tok.string
        else:
            typ = None

        typs.append(typ)

        if p.peek(Predict_tfparg):
            arg = tfparg(p)
            args.append(arg)
        else:
            break

        if (tok := p.peek(TYPE_COMMENT)):
            end = True
            typs.append(tok.string)
            break

    if not end:
        if p.peek(Predict_typedargslist_starargs):
            stargs = typedargslist_starargs(p)
        if p.peek(Predict_typedargslist_kwargs):
            kwargs = typedargslist_kwargs(p)

    return args, typs, stargs, kwargs

Predict_typedargslist_args = Predict_tfparg

def typedargslist_starargs(p):
    '''
        typedargslist_starargs:
                '*' [ tfpdef ]
                ( ',' [ TYPE_COMMENT ] tfparg )*
                (       TYPE_COMMENT
                    |   [ ',' [ TYPE_COMMENT ] [
                                typedargslist_kwargs
                            ]
                        ]
                )
            ;
    '''

    p.expect('*')

    if p.peek(Predict_tfpdef):
        arg = tfpdef(p)
    else:
        arg = None

    args = [ arg ]

    typs = []

    while p.consume(','):
        if (tok := p.peek(TYPE_COMMENT)):
            typ = tok.string
        else:
            typ = None

        typs.append(typ)

        if p.peek(Predict_tfparg):
            arg = tfparg(p)
            args.append(arg)

        else:
            break

    if p.peek(Predict_typedargslist_kwargs):
        kwargs = typedargslist_kwargs(p)
    else:
        kwargs = None

    return args, typs, kwargs

Predict_typedargslist_starargs = { '*' }

def typedargslist_kwargs(p):
    '''
        typedargslist_kwargs:
                '**' tfpdef [ ',' ] [ TYPE_COMMENT ]
            ;
    '''

    p.expect('**')

    defn = tfpdef(p)

    if p.consume(','):
        pass

    if (tok := p.peek(TYPE_COMMENT)):
        typ = tok.string
    else:
        typ = None

    return defn, typ

Predict_typedargslist_kwargs = { '**' }

def typedargslist(p):
    '''
        typedargslist:
                typedargslist_args
            |   typedargslist_starargs
            |   typedargslist_kwargs
            ;
    '''

    if p.peek(Predict_typedargslist_args):
        return typedargslist_args(p)
    elif p.peek(Predict_typedargslist_starargs):
        return typedargslist_starargs(p)
    elif p.peek(Predict_typedargslist_kwargs):
        return typedargslist_kwargs(p)
    else:
        return p.expectedError(Predict_typedargslist)

Predict_typedargslist = Predict_typedargslist_args | Predict_typedargslist_starargs | Predict_typedargslist_kwargs

def parameters(p) -> ast.arguments:
    '''
        parameters:
                '(' [ typedargslist ] ')'
            ;
    '''

    p.expect('(')

    if p.peek(Predict_typedargslist):
        args = typedargslist(p)
    else:
        # FIXME
        args = []

    p.expect(')')

    return args

Predict_parameters = { '(' }

def suite(p) -> List[ast.Stmt]:
    '''
        suite:
                simple_stmt
            |   NEWLINE INDENT stmt+ DEDENT
            ;
    '''

    body: List[ast.Stmt] = []

    if p.consume(NEWLINE):
        p.expect(INDENT)

        while True:
            body += stmt(p)

            if not p.peek(Predict_stmt):
                break

        p.expect(DEDENT)

    elif p.peek(Predict_simple_stmt):
        body += simple_stmt(p)

    return body

Predict_suite = { NEWLINE } | Predict_simple_stmt

def func_body_suite(p):
    # the TYPE_COMMENT in suites is only parsed for funcdefs,
    # but can't go elsewhere due to ambiguity
    '''
        func_body_suite:
                simple_stmt
            |   NEWLINE [ TYPE_COMMENT NEWLINE ] INDENT stmt+ DEDENT
            ;
    '''

    body = []

    tc = None

    if p.consume(NEWLINE):
        if (tok := p.consume(TYPE_COMMENT)):
            tc = tok.string
            p.expect(NEWLINE)

        p.expect(INDENT)

        while True:
            body += stmt(p)

            if not p.peek(Predict_stmt):
                break

        p.expect(DEDENT)

    elif p.peek(Predict_simple_stmt):
        body += simple_stmt(p)

    return tc, body

Predict_func_body_suite = { NEWLINE } | Predict_simple_stmt

def funcdef(p) -> ast.FunctionDef:
    '''
        funcdef:
                'def' NAME parameters [ '->' test ] ':' [ TYPE_COMMENT ] func_body_suite
            ;
    '''

    p.expect('def')

    n = name(p)
    args = parameters(p)

    if p.consume('->'):
        returns = test(p)
    else:
        returns = None

    p.expect(':')

    if (tok := p.peek(TYPE_COMMENT)):
        tc = tok.string
    else:
        tc = None

    _, body = func_body_suite(p)

    return ast.FunctionDef(name=n, args=args, body=body, decorators=[], returns=returns, type_comment=tc)

Predict_funcdef = { 'def' }

def classdef(p) -> ast.ClassDef:
    '''
        classdef:
                'class' NAME [ '(' [ arglist ] ')' ] ':' suite
            ;
    '''

    p.expect('class')
    n = name(p)

    bases = []
    keywords = []

    if p.consume('('):
        if p.peek(Predict_arglist):
            base, keywords = arglist(p)

        p.expect(')')

    p.expect(':')

    body = suite(p)

    return ast.ClassDef(name=n, bases=bases, keywords=keywords, body=body, decorators=[])

Predict_classdef = { 'class' }

def async_funcdef(p) -> ast.AsyncFunctionDef:
    '''
        async_funcdef:
                'async' funcdef
            ;
    '''

    p.expect('async')
    t = funcdef(p)

    return ast.FunctionDef(name=t.name, args=t.args, body=t.body, decorators=t.decorators, returns=t.returns, type_comment=t.type_comment)

Predict_async_funcdef = { 'async' }

def decorator(p):
    '''
        decorator:
                '@' dotted_name [ '(' [ arglist ] ')' ] NEWLINE
            ;
    '''

    p.expect('@')

    n = dotted_name(p)

    bases = []
    keywords = []

    if p.consume('('):
        if p.peek(Predict_arglist):
            base, keywords = arglist(p)

        p.expect(')')

    p.expect(NEWLINE)

    # FIXME:

Predict_decorator = { '@' }

def decorators(p):
    '''
        decorators:
                decorator+
            ;
    '''

    decs = []

    while True:
        dec = decorator(p)
        decs.append(dec)

        if not p.peek(Predict_decorator):
            break

    return decs

Predict_decorators = Predict_decorator

def decorated(p) -> ast.Stmt:
    '''
        decorated:
                decorators ( classdef | funcdef | async_funcdef )
            ;
    '''

    decs = decorators(p)

    if p.peek(Predict_classdef):
        t = classdef(p)
        t.decorators = decs
        return t
    elif p.peek(Predict_funcdef):
        t = funcdef(p)
        t.decorators = decs
        return t
    elif p.peek(Predict_async_funcdef):
        t = async_funcdef(p)
        t.decorators = decs
        return t
    else:
        return p.expectedError(Predict_classdef, Predict_funcdef, Predict_async_funcdef)

Predict_decorated = Predict_decorators

def async_stmt(p) -> ast.Stmt:
    '''
        async_stmt:
                'async' ( funcdef | with_stmt | for_stmt )
            ;
    '''

    p.expect('async')

    if p.peek(Predict_funcdef):
        t = funcdef(p)
        return ast.AsyncFunctionDef(name=t.name, args=t.args, body=t.body, decorators=t.decorators, returns=t.returns, type_comment=t.type_comment)

    elif p.peek(Predict_with_stmt):
        t = with_stmt(p)
        return ast.AsyncWith(items=t.items, body=t.body, type_comment=t.type_comment)

    elif p.peek(Predict_for_stmt):
        t = for_stmt(p)
        return ast.AsyncFor(target=t.target, iter=t.iter, body=t.body, orelse=t.orelse, type_comment=t.type_comment)

    else:
        return p.expectedError(Predict_funcdef, Predict_with_stmt, Predict_for_stmt)

Predict_async_stmt = { 'async' }

def compound_stmt(p) -> ast.Stmt:
    '''
        compound_stmt:
                if_stmt
            |   while_stmt
            |   for_stmt
            |   try_stmt
            |   with_stmt
            |   funcdef
            |   classdef
            |   decorated
            |   async_stmt
            ;
    '''

    if p.peek(Predict_if_stmt):
        return if_stmt(p)
    elif p.peek(Predict_while_stmt):
        return while_stmt(p)
    elif p.peek(Predict_for_stmt):
        return for_stmt(p)
    elif p.peek(Predict_try_stmt):
        return try_stmt(p)
    elif p.peek(Predict_with_stmt):
        return with_stmt(p)
    elif p.peek(Predict_funcdef):
        return funcdef(p)
    elif p.peek(Predict_classdef):
        return classdef(p)
    elif p.peek(Predict_decorated):
        return decorated(p)
    elif p.peek(Predict_async_stmt):
        return async_stmt(p)
    else:
        return p.expectedError(Predict_compound_stmt)

Predict_compound_stmt = Predict_if_stmt | Predict_while_stmt | Predict_for_stmt | Predict_try_stmt | Predict_with_stmt | Predict_funcdef | Predict_classdef | Predict_decorated | Predict_async_stmt

def single_input(p) -> ast.Interactive:
    '''
        single_input:
                NEWLINE
            |   simple_stmt
            |   compound_stmt NEWLINE
            ;
    '''

    body = []

    if p.consume(NEWLINE):
        pass

    elif p.peek(Predict_simple_stmt):
        body = simple_stmt(p)

    elif p.peek(Predict_compound_stmt):
        body = compound_stmt(p)
        p.expect(NEWLINE)

    else:
        return p.expectedError(Predict_simple_stmt, Predict_compound_stmt)

    return ast.Interactive(body=body)

Predict_single_input = { NEWLINE } | Predict_simple_stmt | Predict_compound_stmt

def stmt(p) -> List[ast.Stmt]:
    '''
        stmt:
                simple_stmt
            |   compound_stmt
            ;
    '''

    if p.peek(Predict_simple_stmt):
        return simple_stmt(p)
    elif p.peek(Predict_compound_stmt):
        return [ compound_stmt(p) ]
    else:
        return p.expectedError(Predict_stmt)

Predict_stmt = Predict_simple_stmt | Predict_compound_stmt

def file_input(p) -> ast.Module:
    '''
        file_input:
                ( NEWLINE | stmt )* ENDMARKER
            ;
    '''

    body = []

    while p.peek(NEWLINE, Predict_stmt):
        if p.consume(NEWLINE):
            continue

        body += stmt(p)

    p.expect(ENDMARKER)

    return ast.Module(body=body, type_ignores=[])

Predict_file_input = { NEWLINE, ENDMARKER } | Predict_stmt

def argument(p) -> Tuple[Optional[ast.Expr], Optional[ast.keyword]]:
    '''
        argument:
                test [ comp_for ]
            |   test ':=' test
            |   test '=' test
            |   '**' test
            |   '*' test
    '''

    if p.consume('*'):
        value = test(p)
        return ast.Starred(value=value), None

    elif p.consume('**'):
        value = test(p)
        return None, ast.keyword(arg=None, value=value)

    elif p.peek(Predict_test):
        arg = test(p)

        if p.consume(':='):
            value = test(p)
            return ast.NamedExpr(target=arg, value=value), None

        elif p.consume('='):
            value = test(p)
            return None, ast.keyword(arg=arg, value=value)

        elif p.peek(Predict_comp_for):
            _, comps = comp_for(p)

            return ast.GeneratorExp(elt=arg, generators=comps), None

        else:
            return arg, None

    else:
        return p.expectedError(Predict_argument)

Predict_argument = { '*', '**' } | Predict_test

def arglist(p) -> ArgsList:
    '''
        arglist:
                argument ( ',' argument )* [ ',' ]
            ;
    '''

    args = []
    kwargs = []

    while True:
        arg, kwarg = argument(p)

        if arg is not None:
            args.append(arg)
        elif kwarg is not None:
            kwargs.append(kwarg)

        if not p.consume(','):
            break

        if not p.peek(Predict_argument):
            break

    return args, kwargs

Predict_arglist = Predict_argument

def testlist(p) -> ast.Expr:
    '''
        testlist:
                test ( ',' test )* [ ',' ]
            ;
    '''

    t = test(p)

    if not p.peek(','):
        return t

    elts = [ t ]

    while p.consume(','):
        if not p.peek(Predict_test):
            break

        t = test(p)
        elts.append(t)

    return ast.Tuple(elts=elts)

Predict_testlist = Predict_test

def eval_input(p) -> ast.Expression:
    '''
        eval_input:
                testlist NEWLINE* ENDMARKER
            ;
    '''

    body = testlist(p)

    while p.consume(NEWLINE):
        pass

    p.expect(ENDMARKER)

    return ast.Expression(body=body)

Predict_eval_input = Predict_testlist

def sync_comp_for(p) -> List[ast.comprehension]:
    '''
        sync_comp_for:
                'for' exprlist 'in' or_test [ comp_iter ]
            ;
    '''

    p.expect('for')
    target = exprlist(p)
    p.expect('in')
    iters = or_test(p)
    comp = ast.comprehension(target=target, iter=iters, ifs=[])

    if p.peek(Predict_comp_iter):
        ifs, comps = comp_iter(p)

        comp.ifs = ifs
        comps.insert(0, comp)

    else:
        comps = [ comp ]

    return comps

Predict_sync_comp_for = { 'for' }

CompTail = Tuple[List[ast.Expr], List[ast.comprehension]]

def comp_for(p) -> CompTail:
    '''
        comp_for:
                [ 'async' ] sync_comp_for
            ;
    '''

    if p.consume('async'):
        is_async = True
    else:
        is_async = False

    comps = sync_comp_for(p)

    comps[0].is_async = is_async

    return [], comps

Predict_comp_for = { 'async' } | Predict_sync_comp_for

def comp_if(p) -> CompTail:
    '''
        comp_if:
                'if' test_nocond [ comp_iter ]
            ;
    '''

    p.expect('if')

    t = test_nocond(p)

    if p.peek(Predict_comp_iter):
        ifs, comps = comp_iter(p)
        ifs.insert(0, t)

    else:
        ifs = [ t ]
        comps = []

    return ifs, comps

Predict_comp_if = { 'if' }

def comp_iter(p) -> CompTail:
    '''
        comp_iter:
                comp_for
            |   comp_if
            ;
    '''

    if p.peek(Predict_comp_for):
        return comp_for(p)
    elif p.peek(Predict_comp_if):
        return comp_if(p)
    else:
        return p.expectedError(Predict_comp_for, Predict_comp_if)

Predict_comp_iter = Predict_comp_for | Predict_comp_if
