from io import BytesIO
import token
from tokenize import tokenize
import tokenize as tokens

class CPythonLexer:
    def __init__(self, ts):
        self._ts = ts

    @classmethod
    def from_string(cls, s):
        fobj = BytesIO(s.encode('utf-8'))
        return cls.from_fobj(fobj)

    @classmethod
    def from_fobj(cls, fobj):
        ts = tokenize(fobj.readline)
        return cls(ts)

    def __iter__(self):
        for tok in self._ts:
            if tok.type == token.COMMENT:
                continue

            if tok.type == token.NL:
                tok = tok._replace(type=token.NEWLINE)
                continue

            yield tok

if not True:
    lex = CPythonLex.from_string(source)
    for x in lex:
        print(x)

############

if not True:
    import re

    SPEC = r'''
        (?P<NEWLINE_INDENT> \n [ \t]+  ) |
        (?P<NEWLINE> \n ) |
        (?P<IDENT>  [_a-zA-Z] [_a-zA-Z0-9]* ) |
        (?P<NUMBER> ( 0 | ([-]? [1-9] [0-9]*) )      ) |
        (?P<LT>     <                       ) |
        (?P<EQ>     =                       ) |
        (?P<COLON>   :                       ) |
        (?P<LBRACE> {                       ) |
        (?P<RBRACE> }                       ) |
        (?P<WS>     \s+                     )
        '''

    get_token = re.compile(SPEC, re.VERBOSE).match
    keywords = { 'if', 'else' }
    source = '\n' + source

    while len(source) > 0:
        m = get_token(source)
        if m is None:
            raise Exception(f'Unable to scan: {source!r}')

        start, stop = m.span()
        kind = m.lastgroup
        val = m.group(kind)

        # Find reserved keyword identifiers
        if kind == 'IDENT' and val in keywords:
            kind = val.upper()
            val = ''

        if kind != 'WS':
            print(f'{kind:6} {val}')

        source = source[stop:]
