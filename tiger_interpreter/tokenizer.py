import re
from abc import abstractmethod, ABC
from io import StringIO
from typing import Iterator, Tuple, Union, Optional


class Token(ABC):

    def __getattr__(self, attr_name):
        if attr_name.startswith('is_'):
            cls_name = attr_name[3:].split('_').title().join('')
            return self.__class__.__name__ == cls_name

        raise AttributeError('{} has no attribute {}'.format(self, attr_name))

    @classmethod
    @abstractmethod
    def matches(cls, tkn: str) -> bool:
        raise NotImplementedError

    def __init__(self, token_value: str):
        if not self.matches(token_value):
            raise ValueError(
                '{} is not a valid {}'.format(
                    token_value, self.__class__.__name__
                )
            )
        self.value = token_value

    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value

    def __ne__(self, other):
        return not(self == other)


class _Enum(type):

    def __init__(cls, *args, **kwargs):
        super(_Enum, cls).__init__(*args, **kwargs)
        cls._names = [
            attr_name for attr_name in dir(cls)
            if attr_name.upper() == attr_name and not attr_name.startswith('_')
        ]
        cls._items = {
            attr_name: getattr(cls, attr_name) for attr_name in cls._names
        }
        cls._values = [getattr(cls, attr_name) for attr_name in cls._names]


class _EnumToken(Token, metaclass=_Enum):

    @classmethod
    def matches(cls, tkn: str) -> bool:
        return tkn in cls._values


class Punctuation(_EnumToken):
    PAREN_OPEN = "("
    PAREN_CLOSE = ")"
    BRACKET_OPEN = "["
    BRACKET_CLOSE = "]"
    CURLY_OPEN = "{"
    CURLY_CLOSE = "}"
    COLON = ":"
    ASSIGNMENT = ":="
    DOT = "."
    COMMA = ","
    SEMI_COLON = ";"


class Operator(_EnumToken):
    MUL = "*"
    DIV = "/"
    ADD = "+"
    SUB = "-"
    EQ = "="
    NE = "<>"
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    AND = "&"
    OR = "|"


class Keyword(_EnumToken):
    ARRAY = "array"
    BREAK = "break"
    DO = "do"
    ELSE = "else"
    END = "end"
    FOR = "for"
    FUNCTION = "function"
    IF = "if"
    IN = "in"
    LET = "let"
    NIL = "nil"
    OF = "of"
    THEN = "then"
    TO = "to"
    TYPE = "type"
    VAR = "var"
    WHILE = "while"


class Identifier(Token):
    @classmethod
    def matches(cls, tkn: str) -> bool:
        return bool(re.match(r'[a-zA-Z][a-zA-Z0-9_]*', tkn))


class IntegerLiteral(Token):
    @classmethod
    def matches(cls, tkn: str) -> bool:
        return bool(re.match(r'[0-9]+', tkn))


class StringLiteral(Token):
    @classmethod
    def matches(cls, tkn: str) -> bool:
        return bool(re.match(r'^".*"$', tkn))


class TokenizerException(Exception):
    pass


class Tokenizer(object):
    _BACKSLASH = "\\"
    _DOUBLEQUOTE = '"'

    def __init__(self, program: str):
        self._program = program
        self._tokens = None
        self._next_token_idx = None

    @property
    def _program_without_comments(self) -> str:
        return re.sub(r'/\*.*\*/', self._program, '')

    def next_token(self) -> Optional[Token]:
        if self._tokens is None:
            self._tokens = list(self._yield_tokens())
            self._next_token_idx = 0
        elif self._next_token_idx >= len(self._tokens):
            return None
        else:
            tkn = self._tokens[self._next_token_idx]
            self._next_token_idx += 1
            return tkn

    def rewind(self, n: int = 1):
        if self._next_token_idx - n <= 0:
            raise TokenizerException(
                'Cannot rewind {} tokens, have only read {}'.format(
                    n, self._next_token_idx
                )
            )
        self._next_token_idx -= n

    def _yield_tokens(self) -> Iterator[Token]:
        reader = StringIO(self._program_without_comments)
        char = None
        while char != '':
            char = reader.read(1)

            if re.match(r'\s', char):
                continue

            if char == self._DOUBLEQUOTE:
                token, reader = self._read_string(reader)
                yield token
            elif re.match(r'[a-zA-Z]', char):
                token, reader = self._read_identifier_or_keyword(char, reader)
                yield token
            else:
                token, reader = self._read_punctuation_or_operator(
                    char, reader
                )
                yield token

    def _read_string(self, reader: StringIO) -> Tuple[StringLiteral, StringIO]:
        token_val = ''
        escape_map = {
            '"': '"',
            'n': '\n',
            't': '\t',
            self._BACKSLASH: self._BACKSLASH,
        }
        while True:
            char = reader.read(1)

            if char == self._BACKSLASH:
                next_char = reader.read(1)
                if next_char in escape_map:
                    token_val += escape_map[next_char]
                elif next_char == '^':
                    # TODO: control characters
                    raise NotImplementedError()
                else:
                    ascii_code = next_char
                    for _ in range(2):
                        ascii_code += reader.read(1)

                    if not re.match('[0-9]{3}', next_char):
                        so_far = char + next_char
                        raise TokenizerException(
                            'Invalid escape sequence: {}'.format(so_far)
                        )

                    ascii_code = int(ascii_code)
                    if ascii_code > 127:
                        raise TokenizerException(
                            '{} is outside ASCII range'.format(ascii_code)
                        )

                    token_val += chr(ascii_code)
            elif char == self._DOUBLEQUOTE:
                return StringLiteral(token_val), reader
            else:
                token_val += char

    def _read_identifier_or_keyword(
        self, first_char, reader: StringIO
    ) -> Tuple[Union[Identifier, Keyword], StringIO]:
        """
        Keyword tokens match the same pattern as identifier tokens, so we must
        give preference to keywords (which are, after all, reserved).
        """
        token_val = first_char
        while re.match(r'[a-zA-Z0-9_]', token_val):
            token_val += reader.read(1)

        reader.seek(reader.tell() - 1)
        token_val = token_val[:-1]

        if Keyword.matches(token_val):
            return Keyword(token_val), reader
        else:
            return Identifier(token_val), reader

    def _read_punctuation_or_operator(
        self, first_char, reader: StringIO
    ) -> Tuple[Union[Punctuation, Operator], StringIO]:
        """
        Punctuation and operators can be 1-2 characters. Because some 1-char
        tokens are a prefix of a 2-char token (e.g. "<" and "<>"), we try
        a greedy match on the 2-char first. If that fails, match a 1-char.

        The order of matching punctuation vs operator doesn't matter: none of
        their tokens share a prefix.
        """
        short_token = first_char
        long_token = short_token + reader.read(1)
        if Punctuation.matches(long_token):
            return Punctuation(long_token), reader
        elif Operator.matches(long_token):
            return Operator(long_token), reader
        else:
            reader.seek(reader.tell() - 1)
            if Punctuation.matches(short_token):
                return Punctuation(short_token), reader
            else:
                return Operator(long_token), reader
