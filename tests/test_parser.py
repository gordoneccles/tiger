from pytest import fixture

from tiger_pl.parser import (
    TigerParser,
    Program,
    LetExpression,
    FunDeclaration,
    Identifier,
    CallExpression,
    StringLiteralExpression,
)
from tiger_pl.lexer import TigerLexer


@fixture
def hello_world_lexer():
    let_hello_world = """
    /* Hello-world with function */
    let
        function hello() = print("Hello, World!\n")
    in
        hello()
    end
    """
    return TigerLexer(let_hello_world)


class TestParser(object):
    def test_hello_world(self, hello_world_lexer):
        parser = TigerParser()
        ast = parser.parse(hello_world_lexer)
        assert ast == Program(
            LetExpression(
                [
                    FunDeclaration(
                        Identifier("hello"),
                        [],
                        CallExpression(
                            Identifier("print"),
                            [StringLiteralExpression("Hello, World!\n")],
                        ),
                    )
                ],
                [CallExpression(Identifier("hello"), [])],
            )
        )
