from tiger_interpreter.tokenizer import TigerLexer
from tiger_interpreter.parser import TigerParser
# form tiger_interpreter.interpreter import TigerInterpreter


class Tiger(object):

    def __init__(self, program_file):
        self._prog_file = program_file

    def execute(self):
        prog_data = open(self._prog_file, 'r').read()
        tknizer = TigerLexer(prog_data)
        program = TigerParser().parse(tknizer)
        # TODO
        # interpreter = TigerInterpreter(program)
        # interpreter.execute()
