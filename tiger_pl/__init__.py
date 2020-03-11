from tiger_pl.lexer import TigerLexer
from tiger_pl.parser import TigerParser


class Tiger(object):
    def __init__(self, program_file):
        self._prog_file = program_file

    def execute(self):
        prog_data = open(self._prog_file, "r").read()
        lexer = TigerLexer(prog_data)
        return TigerParser().parse(lexer)
