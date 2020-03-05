import pytest

from tiger_interpreter.tokenizer import (
    Tokenizer, Identifier, Punctuation, StringLiteral, Keyword, Operator,
    TokenizerException)

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
        assert [t for t in Tokenizer(hello_world)._yield_tokens()] == [
            Identifier('print'),
            Punctuation('('),
            StringLiteral('Hello, World!\n'),
            Punctuation(')'),
        ]

    def test_let_hello_world(self):
        assert [
           t for t in Tokenizer(let_hello_world)._yield_tokens()
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
        tokenizer = Tokenizer(hello_world)
        tokens = []
        while True:
            tkn = tokenizer.next_token()
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

    def test_rewind(self):
        tokenizer = Tokenizer(hello_world)
        tokens = [tokenizer.next_token(), tokenizer.next_token()]
        tokenizer.rewind()
        while True:
            tkn = tokenizer.next_token()
            if tkn is None:
                break
            else:
                tokens.append(tkn)

        assert tokens == [
            Identifier('print'),
            Punctuation('('),
            Punctuation('('),
            StringLiteral('Hello, World!\n'),
            Punctuation(')'),
        ]

        with pytest.raises(TokenizerException):
            tokenizer.rewind(10)

    def test_comment_stripping(self):
        tokenizer = Tokenizer(hello_world)
        assert tokenizer._program_without_comments == (
            'print("Hello, World!\n")'
        )

        tokenizer = Tokenizer(very_commented_hello_world)
        assert tokenizer._program_without_comments == (
            'print("Hello, World!\n")'
        )

