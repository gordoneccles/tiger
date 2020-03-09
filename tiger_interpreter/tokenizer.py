import re
from abc import abstractmethod, ABC
from io import StringIO
from typing import Iterator, Tuple, Union, Optional


class _AbstractToken(ABC):

    @classmethod
    @abstractmethod
    def matches(cls, tkn: str) -> bool:
        raise NotImplementedError


class _EnumMeta(type):

    def __init__(cls, *args, **kwargs):
        super(_EnumMeta, cls).__init__(*args, **kwargs)
        cls._names = [
            attr_name for attr_name in dir(cls)
            if attr_name.upper() == attr_name and not attr_name.startswith('_')
        ]
        cls._items = {
            attr_name: getattr(cls, attr_name) for attr_name in cls._names
        }
        cls._values = [getattr(cls, attr_name) for attr_name in cls._names]


class _AbstractEnumToken(metaclass=_EnumMeta):

    @classmethod
    def matches(cls, tkn: str) -> bool:
        return tkn in cls._values


class Token(object):
    def __getattr__(self, attr_name):
        """
        Rather than having to do isinstance(token, SomeToken), this allows
        for token.is_some_token
        """
        if attr_name.startswith('is_'):
            cls_name = attr_name[3:].title().replace('_', '')
            return self.__class__.__name__ == cls_name

        raise AttributeError('{} has no attribute {}'.format(self, attr_name))

    def __init__(self, token_value: str):
        if not self.matches(token_value):
            raise ValueError(
                '"{}" is not a valid {}'.format(
                    token_value, self.__class__.__name__
                )
            )
        self.value = token_value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return type(self) == type(other) and self.value == other.value

    def __ne__(self, other):
        return not(self == other)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.value)


class Punctuation(Token, _AbstractEnumToken):
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


class Operator(Token, _AbstractEnumToken):
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


class Keyword(Token, _AbstractEnumToken):
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


class Identifier(Token, _AbstractToken):
    @classmethod
    def matches(cls, tkn: str) -> bool:
        return bool(re.match(r'[a-zA-Z][a-zA-Z0-9_]*', tkn))


class IntegerLiteral(Token, _AbstractToken):
    @classmethod
    def matches(cls, tkn: str) -> bool:
        return bool(re.match(r'[0-9]+', tkn))


class StringLiteral(Token, _AbstractToken):
    @classmethod
    def matches(cls, tkn: str) -> bool:
        return True


class TokenizerException(Exception):
    pass


class TigerTokenizer(object):
    _BACKSLASH = "\\"
    _DOUBLEQUOTE = '"'

    def __init__(self, program: str):
        self._program = program
        self._tokens = None
        self._next_token_idx = None

    @property
    def _program_without_comments(self) -> str:
        return re.sub(r'/\*.*\*/', '', self._program).strip()

    def next(self) -> Optional[Token]:
        """
        Get next token and advance in token stream.
        If end of stream has been reached, return None.
        """
        if self._tokens is None:
            self._tokens = list(self._yield_tokens())
            self._next_token_idx = 0

        if self._next_token_idx >= len(self._tokens):
            return None

        tkn = self._tokens[self._next_token_idx]
        self._next_token_idx += 1
        return tkn

    def peek(self, n: int = 0) -> Optional[Token]:
        """
        Peek n tokens ahead without advancing stream.
        """
        if self._tokens is None:
            self._tokens = list(self._yield_tokens())
            self._next_token_idx = 0

        peek_idx = self._next_token_idx + n
        if peek_idx >= len(self._tokens):
            return None

        return self._tokens[peek_idx]

    def _yield_tokens(self) -> Iterator[Token]:
        reader = StringIO(self._program_without_comments)
        char = reader.read(1)
        while char != '':
            if re.match(r'\s', char):
                char = reader.read(1)
                continue

            if char == self._DOUBLEQUOTE:
                token = self._read_string(reader)
                yield token
            elif re.match(r'[a-zA-Z]', char):
                token = self._read_identifier_or_keyword(char, reader)
                yield token
            else:
                token = self._read_punctuation_or_operator(char, reader)
                yield token

            char = reader.read(1)

    def _read_string(self, reader: StringIO) -> StringLiteral:
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
                elif next_char == 's':
                    # TODO: whitespace ignore for multiline strings
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
                return StringLiteral(token_val)
            else:
                token_val += char

    def _read_identifier_or_keyword(
        self, char: str, reader: StringIO
    ) -> Union[Identifier, Keyword]:
        """
        Keyword tokens match the same pattern as identifier tokens, so we must
        give preference to keywords (which are, after all, reserved).
        """
        token_val = char
        char = reader.read(1)
        while re.match(r'[a-zA-Z0-9_]', char):
            token_val += char
            char = reader.read(1)

        if char != '':  # not end of file
            reader.seek(reader.tell() - 1)

        if Keyword.matches(token_val):
            return Keyword(token_val)
        else:
            return Identifier(token_val)

    def _read_punctuation_or_operator(
        self, first_char, reader: StringIO
    ) -> Union[Punctuation, Operator]:
        """
        Punctuation and operators can be 1-2 characters. Because some 1-char
        tokens are a prefix of a 2-char token (e.g. "<" and "<>"), we try
        a greedy match on the 2-char first. If that fails, match a 1-char.

        The order of matching punctuation vs operator doesn't matter: none of
        their tokens share a prefix.
        """
        short_token = first_char
        next_char = reader.read(1)
        long_token = short_token + next_char
        if Punctuation.matches(long_token):
            return Punctuation(long_token)
        elif Operator.matches(long_token):
            return Operator(long_token)
        else:
            if next_char != '':  # not end of file
                reader.seek(reader.tell() - 1)
            if Punctuation.matches(short_token):
                return Punctuation(short_token)
            else:
                return Operator(short_token)
