from pytest import fixture

from tiger_interpreter.parser import TigerParser
from tiger_interpreter.tokenizer import TigerLexer


@fixture
def hello_world_tokenizer():
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

    def test_hello_world(self, hello_world_tokenizer):
        parser = TigerParser()
        ast = parser.parse(hello_world_tokenizer)
