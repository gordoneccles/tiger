import pytest

from tiger_interpreter.lexer import (
    TigerLexer, Identifier, Punctuation, StringLiteral, Keyword, Operator,
    TokenizerException
)

hello_world = """
/* Hello-world */
print("Hello, World!\n")
"""

let_hello_world = """
/* Hello-world with function */
let
    function hello() = print("Hello, World!\n")
in
    hello()
end
"""

very_commented_hello_world = """
/* Hello-world */
print("Hello, World!\n") /* **this/does/the/thing** */
/* cya */
"""


class TestTokenizer(object):
    def test_hello_world(self):
        assert [t for t in TigerLexer(hello_world)._yield_tokens()] == [
            Identifier('print'),
            Punctuation('('),
            StringLiteral('Hello, World!\n'),
            Punctuation(')'),
        ]

    def test_let_hello_world(self):
        assert [
           t for t in TigerLexer(let_hello_world)._yield_tokens()
        ] == [
            Keyword('let'),
            Keyword('function'),
            Identifier('hello'),
            Punctuation('('),
            Punctuation(')'),
            Operator('='),
            Identifier('print'),
            Punctuation('('),
            StringLiteral('Hello, World!\n'),
            Punctuation(')'),
            Keyword('in'),
            Identifier('hello'),
            Punctuation('('),
            Punctuation(')'),
            Keyword('end'),
        ]

    def test_next_token(self):
        tokenizer = TigerLexer(hello_world)
        tokens = []
        while True:
            tkn = tokenizer.next()
            if tkn is None:
                break
            else:
                tokens.append(tkn)

        assert tokens == [
            Identifier('print'),
            Punctuation('('),
            StringLiteral('Hello, World!\n'),
            Punctuation(')'),
        ]

    def test_pee(self):
        tokenizer = TigerLexer(hello_world)
        assert tokenizer.peek() == Identifier('print')
        assert tokenizer.peek(1) == Punctuation('(')
        assert tokenizer.peek(n=3) == Punctuation(')')
        assert tokenizer.peek(n=4) is None
        tokens = []
        while True:
            tkn = tokenizer.next()
            if tkn is None:
                break
            else:
                tokens.append(tkn)

        assert tokens == [
            Identifier('print'),
            Punctuation('('),
            StringLiteral('Hello, World!\n'),
            Punctuation(')'),
        ]
        assert tokenizer.peek() is None

    def test_comment_stripping(self):
        tokenizer = TigerLexer(hello_world)
        assert tokenizer._program_without_comments == (
            'print("Hello, World!\n")'
        )

        tokenizer = TigerLexer(very_commented_hello_world)
        assert tokenizer._program_without_comments == (
            'print("Hello, World!\n")'
        )

