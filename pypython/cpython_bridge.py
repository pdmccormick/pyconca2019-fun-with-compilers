import ast as C

from .ast import *

class CPythonAstTransformer(NodeTransformer):
    def transform(self, node: AST, *args, **kwargs) -> C.AST:
        return self.visit(node, *args, **kwargs)

    def visit_AnnAssign(self, node: AnnAssign, *args, **kwargs) -> C.AnnAssign:
        target = self.visit(node.target, *args, **kwargs)
        annotation = self.visit(node.annotation, *args, **kwargs)
        value = self.visit(node.value, *args, **kwargs)
        simple = self.visit(node.simple, *args, **kwargs)

        return C.AnnAssign(
                target=target,
                annotation=annotation,
                value=value,
                simple=simple,
                )

    def visit_arg(self, node: arg, *args, **kwargs) -> C.arg:
        arg = self.visit(node.arg, *args, **kwargs)
        annotation = self.visit(node.annotation, *args, **kwargs)
        type_comment = self.visit(node.type_comment, *args, **kwargs)

        return C.arg(
                arg=arg,
                annotation=annotation,
                type_comment=type_comment,
                )

    def visit_Assert(self, node: Assert, *args, **kwargs) -> C.Assert:
        test = self.visit(node.test, *args, **kwargs)
        msg = self.visit(node.msg, *args, **kwargs)

        return C.Assert(
                test=test,
                msg=msg,
                )

    def visit_Assign(self, node: Assign, *args, **kwargs) -> C.Assign:
        targets = self.visit(node.targets, *args, **kwargs)
        value = self.visit(node.value, *args, **kwargs)
        type_comment = self.visit(node.type_comment, *args, **kwargs)

        return C.Assign(
                targets=targets,
                value=value,
                type_comment=type_comment,
                )

    def visit_AsyncFor(self, node: AsyncFor, *args, **kwargs) -> C.AsyncFor:
        target = self.visit(node.target, *args, **kwargs)
        iter = self.visit(node.iter, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)
        orelse = self.visit(node.orelse, *args, **kwargs)
        type_comment = self.visit(node.type_comment, *args, **kwargs)

        return C.AsyncFor(
                target=target,
                iter=iter,
                body=body,
                orelse=orelse,
                type_comment=type_comment,
                )

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef, *args, **kwargs) -> C.AsyncFunctionDef:
        name = self.visit(node.name, *args, **kwargs)
        defargs = self.visit(node.args, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)
        decorators = self.visit(node.decorators, *args, **kwargs)
        returns = self.visit(node.returns, *args, **kwargs)
        type_comment = self.visit(node.type_comment, *args, **kwargs)

        return C.AsyncFunctionDef(
                name=name,
                args=defargs,
                body=body,
                decorator_list=decorators,
                returns=returns,
                type_comment=type_comment,
                )

    def visit_AsyncWith(self, node: AsyncWith, *args, **kwargs) -> C.AsyncWith:
        items = self.visit(node.items, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)
        type_comment = self.visit(node.type_comment, *args, **kwargs)

        return C.AsyncWith(
                items=items,
                body=body,
                type_comment=type_comment,
                )

    def visit_Attribute(self, node: Attribute, *args, **kwargs) -> C.Attribute:
        value = self.visit(node.value, *args, **kwargs)
        attr = self.visit(node.attr, *args, **kwargs)

        return C.Attribute(
                value=value,
                attr=attr,
                )

    def visit_AugAssign(self, node: AugAssign, *args, **kwargs) -> C.AugAssign:
        target = self.visit(node.target, *args, **kwargs)
        op = self.visit(node.op, *args, **kwargs)
        value = self.visit(node.value, *args, **kwargs)

        return C.AugAssign(
                target=target,
                op=op,
                value=value,
                )

    def visit_Await(self, node: Await, *args, **kwargs) -> C.Await:
        value = self.visit(node.value, *args, **kwargs)

        return C.Await(
                value=value,
                )

    def visit_BinOp(self, node: BinOp, *args, **kwargs) -> C.BinOp:
        left = self.visit(node.left, *args, **kwargs)
        op = self.visit(node.op, *args, **kwargs)
        right = self.visit(node.right, *args, **kwargs)

        return C.BinOp(
                left=left,
                op=op,
                right=right,
                )

    def visit_BoolOp(self, node: BoolOp, *args, **kwargs) -> C.BoolOp:
        op = self.visit(node.op, *args, **kwargs)
        values = self.visit(node.values, *args, **kwargs)

        return C.BoolOp(
                op=op,
                values=values,
                )

    def visit_Break(self, node: Break, *args, **kwargs) -> C.Break:
        return C.Break()

    def visit_Call(self, node: Call, *args, **kwargs) -> C.Call:
        func = self.visit(node.func, *args, **kwargs)
        callargs = self.visit(node.args, *args, **kwargs)
        keywords = self.visit(node.keywords, *args, **kwargs)

        return C.Call(
                func=func,
                args=callargs,
                keywords=keywords,
                )

    def visit_ClassDef(self, node: ClassDef, *args, **kwargs) -> C.ClassDef:
        name = self.visit(node.name, *args, **kwargs)
        bases = self.visit(node.bases, *args, **kwargs)
        keywords = self.visit(node.keywords, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)
        decorators = self.visit(node.decorators, *args, **kwargs)

        return C.ClassDef(
                name=name,
                bases=bases,
                keywords=keywords,
                body=body,
                decorators=decorators,
                )

    def visit_Compare(self, node: Compare, *args, **kwargs) -> C.Compare:
        left = self.visit(node.left, *args, **kwargs)
        ops = self.visit(node.ops, *args, **kwargs)
        comparators = self.visit(node.comparators, *args, **kwargs)

        return C.Compare(
                left=left,
                ops=ops,
                comparators=comparators,
                )

    def visit_Constant(self, node: Constant, *args, **kwargs) -> C.Constant:
        value = self.visit(node.value, *args, **kwargs)
        kind = self.visit(node.kind, *args, **kwargs)

        return C.Constant(
                value=value,
                kind=kind,
                )

    def visit_Continue(self, node: Continue, *args, **kwargs) -> C.Continue:
        return C.Continue()

    def visit_Delete(self, node: Delete, *args, **kwargs) -> C.Delete:
        targets = self.visit(node.targets, *args, **kwargs)

        return C.Delete(
                targets=targets,
                )

    def visit_Dict(self, node: Dict, *args, **kwargs) -> C.Dict:
        keys = self.visit(node.keys, *args, **kwargs)
        values = self.visit(node.values, *args, **kwargs)

        return C.Dict(
                keys=keys,
                values=values,
                )

    def visit_DictComp(self, node: DictComp, *args, **kwargs) -> C.DictComp:
        key = self.visit(node.key, *args, **kwargs)
        value = self.visit(node.value, *args, **kwargs)
        generators = self.visit(node.generators, *args, **kwargs)

        return C.DictComp(
                key=key,
                value=value,
                generators=generators,
                )

    def visit_ExceptHandler(self, node: ExceptHandler, *args, **kwargs) -> C.ExceptHandler:
        typ = self.visit(node.typ, *args, **kwargs)
        name = self.visit(node.name, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)

        return C.ExceptHandler(
                typ=typ,
                name=name,
                body=body,
                )

    def visit_Expr(self, node: Expr, *args, **kwargs) -> C.Expr:
        return C.Expr()

    def visit_ExprStmt(self, node: ExprStmt, *args, **kwargs) -> C.Expr:
        value = self.visit(node.value, *args, **kwargs)

        return C.Expr(
                value=value,
                )

    def visit_Expression(self, node: Expression, *args, **kwargs) -> C.Expression:
        body = self.visit(node.body, *args, **kwargs)

        return C.Expression(
                body=body,
                )

    def visit_ExtSlice(self, node: ExtSlice, *args, **kwargs) -> C.ExtSlice:
        dims = self.visit(node.dims, *args, **kwargs)

        return C.ExtSlice(
                dims=dims,
                )




    ###############################################################3

    # DEMO

    def visit_For(self, node: For, *args, **kwargs) -> C.For:
        target = self.visit(node.target, *args, **kwargs)
        iter = self.visit(node.iter, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)
        cond = self.visit(node.cond, *args, **kwargs)
        orelse = self.visit(node.orelse, *args, **kwargs)
        type_comment = self.visit(node.type_comment, *args, **kwargs)

        if cond is not None:
            check = C.If(
                    test=C.UnaryOp(op=C.Not(), operand=cond),
                    body=[ C.Break() ],
                    orelse=[],
                    )

            body = [ check ] + body

        return C.For(
                target=target,
                iter=iter,
                body=body,
                orelse=orelse,
                type_comment=type_comment,
                )

    ###############################################################3





    '''
    def visit_For(self, node: For, *args, **kwargs) -> C.For:
        target = self.visit(node.target, *args, **kwargs)
        iter = self.visit(node.iter, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)
        orelse = self.visit(node.orelse, *args, **kwargs)
        type_comment = self.visit(node.type_comment, *args, **kwargs)

        return C.For(
                target=target,
                iter=iter,
                body=body,
                orelse=orelse,
                type_comment=type_comment,
                )
    '''

    def visit_FormattedValue(self, node: FormattedValue, *args, **kwargs) -> C.FormattedValue:
        value = self.visit(node.value, *args, **kwargs)
        conversion = self.visit(node.conversion, *args, **kwargs)
        format_spec = self.visit(node.format_spec, *args, **kwargs)

        return C.FormattedValue(
                value=value,
                conversion=conversion,
                format_spec=format_spec,
                )

    def visit_FunctionDef(self, node: FunctionDef, *args, **kwargs) -> C.FunctionDef:
        name = self.visit(node.name, *args, **kwargs)
        defargs = self.visit(node.args, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)
        decorators = self.visit(node.decorators, *args, **kwargs)
        returns = self.visit(node.returns, *args, **kwargs)
        type_comment = self.visit(node.type_comment, *args, **kwargs)

        return C.FunctionDef(
                name=name,
                args=defargs,
                body=body,
                decorator_list=decorators,
                returns=returns,
                type_comment=type_comment,
                )

    def visit_GeneratorExp(self, node: GeneratorExp, *args, **kwargs) -> C.GeneratorExp:
        elt = self.visit(node.elt, *args, **kwargs)
        generators = self.visit(node.generators, *args, **kwargs)

        return C.GeneratorExp(
                elt=elt,
                generators=generators,
                )

    def visit_Global(self, node: Global, *args, **kwargs) -> C.Global:
        names = self.visit(node.names, *args, **kwargs)

        return C.Global(
                names=names,
                )

#   def visit_Ident(self, node: Ident, *args, **kwargs) -> C.Ident:
#       value = self.visit(node.value, *args, **kwargs)
#
#       return C.Ident(
#               value=value,
#               )

    def visit_If(self, node: If, *args, **kwargs) -> C.If:
        test = self.visit(node.test, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)
        orelse = self.visit(node.orelse, *args, **kwargs)

        return C.If(
                test=test,
                body=body,
                orelse=orelse,
                )

    def visit_IfExpr(self, node: IfExpr, *args, **kwargs) -> C.IfExp:
        test = self.visit(node.test, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)
        orelse = self.visit(node.orelse, *args, **kwargs)

        return C.IfExp(
                test=test,
                body=body,
                orelse=orelse,
                )

    def visit_Import(self, node: Import, *args, **kwargs) -> C.Import:
        names = self.visit(node.names, *args, **kwargs)

        return C.Import(
                names=names,
                )

    def visit_ImportFrom(self, node: ImportFrom, *args, **kwargs) -> C.ImportFrom:
        module = self.visit(node.module, *args, **kwargs)
        names = self.visit(node.names, *args, **kwargs)
        level = self.visit(node.level, *args, **kwargs)

        return C.ImportFrom(
                module=module,
                names=names,
                level=level,
                )

    def visit_Index(self, node: Index, *args, **kwargs) -> C.Index:
        value = self.visit(node.value, *args, **kwargs)

        return C.Index(
                value=value,
                )

    def visit_Interactive(self, node: Interactive, *args, **kwargs) -> C.Interactive:
        body = self.visit(node.body, *args, **kwargs)

        return C.Interactive(
                body=body,
                )

    def visit_JoinedStr(self, node: JoinedStr, *args, **kwargs) -> C.JoinedStr:
        values = self.visit(node.values, *args, **kwargs)

        return C.JoinedStr(
                values=values,
                )

    def visit_Lambda(self, node: Lambda, *args, **kwargs) -> C.Lambda:
        defargs = self.visit(node.args, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)

        return C.Lambda(
                args=defargs,
                body=body,
                )

    def visit_ListComp(self, node: ListComp, *args, **kwargs) -> C.ListComp:
        elt = self.visit(node.elt, *args, **kwargs)
        generators = self.visit(node.generators, *args, **kwargs)

        return C.ListComp(
                elt=elt,
                generators=generators,
                )

    def visit_ListExpr(self, node: ListExpr, *args, **kwargs) -> C.List:
        elts = self.visit(node.elts, *args, **kwargs)

        return C.List(
                elts=elts,
                )

    def visit_Module(self, node: Module, *args, **kwargs) -> C.Module:
        body = self.visit(node.body, *args, **kwargs)
        type_ignores = self.visit(node.type_ignores, *args, **kwargs)

        return C.Module(
                body=body,
                type_ignores=type_ignores,
                )

    def visit_Name(self, node: Name, *args, **kwargs) -> C.Name:
        id = self.visit(node.id, *args, **kwargs)

        return C.Name(
                id=id,
                )

    def visit_NamedExpr(self, node: NamedExpr, *args, **kwargs) -> C.NamedExpr:
        target = self.visit(node.target, *args, **kwargs)
        value = self.visit(node.value, *args, **kwargs)

        return C.NamedExpr(
                target=target,
                value=value,
                )

    def visit_Nonlocal(self, node: Nonlocal, *args, **kwargs) -> C.Nonlocal:
        names = self.visit(node.names, *args, **kwargs)

        return C.Nonlocal(
                names=names,
                )

    def visit_NormalSlice(self, node: NormalSlice, *args, **kwargs) -> C.Slice:
        lower = self.visit(node.lower, *args, **kwargs)
        upper = self.visit(node.upper, *args, **kwargs)
        step = self.visit(node.step, *args, **kwargs)

        return C.Slice(
                lower=lower,
                upper=upper,
                step=step,
                )

    def visit_Pass(self, node: Pass, *args, **kwargs) -> C.Pass:
        return C.Pass()

    def visit_Raise(self, node: Raise, *args, **kwargs) -> C.Raise:
        exc = self.visit(node.exc, *args, **kwargs)
        cause = self.visit(node.cause, *args, **kwargs)

        return C.Raise(
                exc=exc,
                cause=cause,
                )

    def visit_Return(self, node: Return, *args, **kwargs) -> C.Return:
        value = self.visit(node.value, *args, **kwargs)

        return C.Return(
                value=value,
                )

    def visit_Set(self, node: Set, *args, **kwargs) -> C.Set:
        elts = self.visit(node.elts, *args, **kwargs)

        return C.Set(
                elts=elts,
                )

    def visit_SetComp(self, node: SetComp, *args, **kwargs) -> C.SetComp:
        elt = self.visit(node.elt, *args, **kwargs)
        generators = self.visit(node.generators, *args, **kwargs)

        return C.SetComp(
                elt=elt,
                generators=generators,
                )

    def visit_Slice(self, node: Slice, *args, **kwargs) -> C.Slice:
        return C.Slice()

    def visit_Starred(self, node: Starred, *args, **kwargs) -> C.Starred:
        value = self.visit(node.value, *args, **kwargs)

        return C.Starred(
                value=value,
                )

    def visit_Subscript(self, node: Subscript, *args, **kwargs) -> C.Subscript:
        value = self.visit(node.value, *args, **kwargs)
        slyce = self.visit(node.slice, *args, **kwargs)

        return C.Subscript(
                value=value,
                slice=slyce,
                )

    def visit_Try(self, node: Try, *args, **kwargs) -> C.Try:
        body = self.visit(node.body, *args, **kwargs)
        handlers = self.visit(node.handlers, *args, **kwargs)
        orelse = self.visit(node.orelse, *args, **kwargs)
        finalbody = self.visit(node.finalbody, *args, **kwargs)

        return C.Try(
                body=body,
                handlers=handlers,
                orelse=orelse,
                finalbody=finalbody,
                )

    def visit_Tuple(self, node: Tuple, *args, **kwargs) -> C.Tuple:
        elts = self.visit(node.elts, *args, **kwargs)

        return C.Tuple(
                elts=elts,
                )

    def visit_TypeIgnore(self, node: TypeIgnore, *args, **kwargs) -> C.TypeIgnore:
        lineno = self.visit(node.lineno, *args, **kwargs)
        tag = self.visit(node.tag, *args, **kwargs)

        return C.TypeIgnore(
                lineno=lineno,
                tag=tag,
                )

    def visit_UnaryOp(self, node: UnaryOp, *args, **kwargs) -> C.UnaryOp:
        op = self.visit(node.op, *args, **kwargs)
        operand = self.visit(node.operand, *args, **kwargs)

        return C.UnaryOp(
                op=op,
                operand=operand,
                )

    def visit_While(self, node: While, *args, **kwargs) -> C.While:
        test = self.visit(node.test, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)
        orelse = self.visit(node.orelse, *args, **kwargs)

        return C.While(
                test=test,
                body=body,
                orelse=orelse,
                )

    def visit_With(self, node: With, *args, **kwargs) -> C.With:
        items = self.visit(node.items, *args, **kwargs)
        body = self.visit(node.body, *args, **kwargs)
        type_comment = self.visit(node.type_comment, *args, **kwargs)

        return C.With(
                items=items,
                body=body,
                type_comment=type_comment,
                )

    def visit_Yield(self, node: Yield, *args, **kwargs) -> C.Yield:
        value = self.visit(node.value, *args, **kwargs)

        return C.Yield(
                value=value,
                )

    def visit_YieldFrom(self, node: YieldFrom, *args, **kwargs) -> C.YieldFrom:
        value = self.visit(node.value, *args, **kwargs)

        return C.YieldFrom(
                value=value,
                )

    def visit_alias(self, node: alias, *args, **kwargs) -> C.alias:
        name = self.visit(node.name, *args, **kwargs)
        asname = self.visit(node.asname, *args, **kwargs)

        return C.alias(
                name=name,
                asname=asname,
                )

    def visit_arguments(self, node: arguments, *args, **kwargs) -> C.arguments:
        posonlyargs = self.visit(node.posonlyargs, *args, **kwargs)
        givenargs = self.visit(node.args, *args, **kwargs)
        varargs = self.visit(node.varargs, *args, **kwargs)
        kwonlyargs = self.visit(node.kwonlyargs, *args, **kwargs)
        kw_defaults = self.visit(node.kw_defaults, *args, **kwargs)
        kwarg = self.visit(node.kwarg, *args, **kwargs)
        defaults = self.visit(node.defaults, *args, **kwargs)

        return C.arguments(
                posonlyargs=posonlyargs,
                args=givenargs,
                varargs=varargs,
                kwonlyargs=kwonlyargs,
                kw_defaults=kw_defaults,
                kwarg=kwarg,
                defaults=defaults,
                )

    def visit_comprehension(self, node: comprehension, *args, **kwargs) -> C.comprehension:
        target = self.visit(node.target, *args, **kwargs)
        iter = self.visit(node.iter, *args, **kwargs)
        ifs = self.visit(node.ifs, *args, **kwargs)
        is_async = self.visit(node.is_async, *args, **kwargs)

        return C.comprehension(
                target=target,
                iter=iter,
                ifs=ifs,
                is_async=is_async,
                )

    def visit_keyword(self, node: keyword, *args, **kwargs) -> C.keyword:
        arg = self.visit(node.arg, *args, **kwargs)
        value = self.visit(node.value, *args, **kwargs)

        return C.keyword(
                arg=arg,
                value=value,
                )

    def visit_withitem(self, node: withitem, *args, **kwargs) -> C.withitem:
        context_expr = self.visit(node.context_expr, *args, **kwargs)
        optional_vars = self.visit(node.optional_vars, *args, **kwargs)

        return C.withitem(
                #context_expr=context_expr,
                optional_vars=optional_vars,
                )

    def visit_UnaryOper(self, node: UnaryOper, *args, **kwargs) -> C.unaryop:
        if node == UnaryOper.Invert:
            return C.Invert
        elif node == UnaryOper.Not:
            return C.Not()
        elif node == UnaryOper.UAdd:
            return C.UAdd
        elif node == UnaryOper.USub:
            return C.USub
        else:
            raise Exception(f'unknown UnaryOper {node!r}')

    def visit_Operator(self, node: Operator, *args, **kwargs) -> C.operator:
        if node == Operator.Add:
            return C.Add()
        elif node == Operator.Sub:
            return C.Sub()
        elif node == Operator.Mult:
            return C.Mult()
        elif node == Operator.MatMult:
            return C.MatMult()
        elif node == Operator.Div:
            return C.Div()
        elif node == Operator.Mod:
            return C.Mod()
        elif node == Operator.Pow:
            return C.Pow()
        elif node == Operator.LShift:
            return C.LShift()
        elif node == Operator.RShift:
            return C.RShift()
        elif node == Operator.BitOr:
            return C.BitOr()
        elif node == Operator.BitXor:
            return C.BitXor()
        elif node == Operator.BitAnd:
            return C.BitAnd()
        elif node == Operator.FloorDiv:
            return C.FloorDiv()
        else:
            raise Exception(f'unknown Operator {node!r}')

    def visit_CmpOp(self, node: CmpOp, *args, **kwargs) -> C.cmpop:
        if node == CmpOp.Eq:
            return C.Eq()
        elif node == CmpOp.NotEq:
            return C.NotEq()
        elif node == CmpOp.Lt:
            return C.Lt()
        elif node == CmpOp.Lte:
            return C.Lte()
        elif node == CmpOp.Gt:
            return C.Gt()
        elif node == CmpOp.Gte:
            return C.Gte()
        elif node == CmpOp.Is:
            return C.Is()
        elif node == CmpOp.IsNot:
            return C.IsNot()
        elif node == CmpOp.In:
            return C.In()
        elif node == CmpOp.NotIn:
            return C.NotIn()
        else:
            raise Exception(f'unknown CmpOp {node!r}')

    def visit_BoolOper(self, node: BoolOper, *args, **kwargs) -> C.boolop:
        if node == BoolOper.And:
            return C.And()
        elif node == BoolOper.Or:
            return C.Or()
        else:
            raise Exception(f'unknown BoolOper {node!r}')
