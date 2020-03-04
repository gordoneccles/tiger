from tiger_interpreter.tokenizer import Tokenizer

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


class TestTokenizer(object):
    def test_hello_world(self):
        assert [t.value for t in Tokenizer(hello_world)._yield_tokens()] == [
            'print', '(', 'Hello, World!\n', ')'
        ]

    def test_let_hello_world(self):
        assert [
           t.value for t in Tokenizer(let_hello_world)._yield_tokens()
        ] == [
            'let',
            'function',
            'hello',
            '(',
            ')',
            '=',
            'print',
            '(',
            'Hello, World!\n',
            ')',
            'in',
            'hello',
            '(',
            ')',
            'end',
        ]
