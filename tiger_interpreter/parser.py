from abc import ABC
from typing import List, Union, Optional

from tiger_interpreter.tokenizer import (
    Token, Keyword, Tokenizer, Punctuation, Operator,
)

"""
program → exp
dec → tyDec | varDec | funDec
tyDec → type tyId = ty
ty → tyId | arrTy | recTy
arrTy → array of tyId
recTy → { fieldDec∗, }
fieldDec → id : tyId
funDec → function id ( fieldDec∗, ) = exp
        | function id ( fieldDec∗, ) : tyId = exp
varDec → var id := exn
        | var id : tyId := exp
lValue → id | subscript | fieldExp
subscript → lValue [ exp ]
fieldExp → lValue . id
exp → lValue | nil | intLit | stringLit
    | seqExp | negation | callExp | infixExp
    | arrCreate | recCreate | assignment
    | ifThenElse | ifThen | whileExp | forExp
    | break | letExp
seqExp → ( exp∗; )
negation → - exp
callExp → id ( exp∗, )
infixExp → exp infixOp exp
arrCreate → tyId [ exp ] of exp
recCreate → tyId { fieldCreate∗, }
fieldCreate → id = exp
assignment → lValue := exp
ifThenElse → if exp then exp else exp
ifThen → if exp then exp
whileExp → while exp do exp
forExp → for id := exp to exp do exp
letExp → let dec+ in exp∗; end
"""


class ParserException(Exception):
    pass


class AbstractSyntaxTreeNode(ABC):
    pass


class IdentifierNode(AbstractSyntaxTreeNode):
    def __init__(self, value: str):
        self.value = value


class TypeIdentifierNode(AbstractSyntaxTreeNode):
    def __init__(self, value: str):
        self.value = value


class Declaration(AbstractSyntaxTreeNode):
    pass


class Expression(AbstractSyntaxTreeNode):
    pass


class LetExpNode(Expression):

    def __init__(
       self, declarations: List[Declaration], expressions: List[Expression]
    ):
        if len(declarations) == 0:
            raise ParserException(
                'Let expressions require at least one declaration.'
            )
        self.declarations = declarations
        self.expressions = expressions


class NilExpNode(Expression):
    pass


class IntegerLiteralExpNode(Expression):

    def __init__(self, integer_val: int):
        self.value = integer_val


class StringLiteralExpNode(Expression):

    def __init__(self, string_val: str):
        self.value = string_val


class TigerType(AbstractSyntaxTreeNode):
    pass


class ArrayTypeNode(TigerType):
    pass


class RecordTypeNode(TigerType):
    pass


class TypeDecNode(Declaration):
    def __init__(
        self,
        type_identifier: TypeIdentifierNode,
        tiger_type: TigerType,
    ):
        self.type_identifier = type_identifier
        self.tiger_type = tiger_type


class VarDecNode(Declaration):

    def __init__(
        self,
        identifier: IdentifierNode,
        expression: Expression,
        type_identifier: Optional[TypeIdentifierNode] = None,
    ):
        self.identifier = identifier
        self.expression = expression
        self.type_identifier = type_identifier


class FieldDecNode(Declaration):
    def __init__(
        self, identifier: IdentifierNode, type_identifier: TypeIdentifierNode
    ):
        self.identifier = identifier
        self.type_identifier = type_identifier


class FuncDecNode(Declaration):
    def __init__(
        self,
        identifier: IdentifierNode,
        field_declarations: List[FieldDecNode],
        expression: Expression,
        type_identifier: Optional[TypeIdentifierNode] = None,
    ):
        self.identifier = identifier
        self.field_declarations = field_declarations,
        self.expression = expression
        self.type_identifier = type_identifier

"""
# Class Heirarchy
AbstractSyntaxTreeNode
    IdentifierNode
    TypeIdentifierNode
    Expression
        LetExpNode
        NilExpNode
        IntegerLiteralNode
        StringLiteralNode
    Declaration
        TypeDecNode
        VarDecNode
        FieldDecNode
        FuncDecNode
    TigerType
        ArrayType
        RecordType
"""


def _assert_tkn_val(token: Token, expect_val: Union[str, List[str]]):
    if isinstance(expect_val, list):
        if token.value not in expect_val:
            raise ParserException(
                'Expected one of {}, found "{}"'.format(
                    expect_val, token.value
                )
            )
    else:
        if token.value != expect_val:
            raise ParserException(
                'Expected "{}", found "{}"'.format(expect_val, token.value)
            )


def _assert_identifier(token: Token):
    if not token.is_identifier:
        raise ParserException(
            'Expected identifier, found {} of value "{}"'.format(
                type(token), token.value
            )
        )


class Parser(object):

    @staticmethod
    def _get_type_annotation(
            tokenizer: Tokenizer
    ) -> Optional[TypeIdentifierNode]:
        if tokenizer.next_token().value == Punctuation.COLON:
            type_id_token = tokenizer.next_token()
            _assert_identifier(type_id_token)
            return TypeIdentifierNode(type_id_token.value)
        else:
            tokenizer.rewind()
            return None

    def __init__(self):
        self._root = None

    def parse(self, tokenizer: Tokenizer):
        # TODO: support programs not wrapped in a let
        self._root = self._parse_let_expression(tokenizer)

    def _parse_expression(self, tokenizer: Tokenizer) -> Expression:
        next_token = tokenizer.next_token()
        if next_token.value == Keyword.NIL:
            return NilExpNode()
        elif next_token.is_integer_literal:
            return IntegerLiteralExpNode(next_token.value)
        elif next_token.is_string_literal:
            return StringLiteralExpNode(next_token.value)
        else:
            raise NotImplementedError('TODO')

    def _parse_let_expression(self, tokenizer: Tokenizer) -> LetExpNode:
        _assert_tkn_val(tokenizer.next_token(), Keyword.LET)
        next_token = tokenizer.next_token()
        decs = []
        tokenizer.rewind()
        while next_token in [Keyword.TYPE, Keyword.FUNCTION, Keyword.VAR]:
            if next_token.value == Keyword.TYPE:
                dec_node = self._parse_type_declaration(tokenizer)
            elif next_token.value == Keyword.FUNCTION:
                dec_node = self._parse_func_declaration(tokenizer)
            else:
                dec_node = self._parse_var_declaration(tokenizer)
            decs.append(dec_node)

        _assert_tkn_val(tokenizer.next_token(), Keyword.IN)

        next_token = tokenizer.next_token()
        exps = []
        while next_token.value != Keyword.END:
            exps.append(self._parse_expression(tokenizer))
            _assert_tkn_val(tokenizer.next_token(), Punctuation.SEMI_COLON)
            next_token = tokenizer.next_token()

        return LetExpNode(decs, exps)

    def _parse_type_declaration(self, tokenizer: Tokenizer) -> TypeDecNode:
        _assert_tkn_val(tokenizer.next_token(), Keyword.TYPE)

        id_token = tokenizer.next_token()
        _assert_identifier(id_token)

        _assert_tkn_val(tokenizer.next_token(), Operator.EQ)

        next_token = tokenizer.next_token()
        tokenizer.rewind()
        if next_token.is_identifier:
            type_node = TypeIdentifierNode(next_token.value)
        elif next_token == Keyword.ARRAY:
            type_node = self._parse_array_type(tokenizer)
        elif next_token == Punctuation.CURLY_OPEN:
            type_node = self._parse_record_type(tokenizer)
        else:
            raise ParserException(
                'Expected type identifier, array type, or record type. '
                'Found "{}"'.format(next_token.value)
            )
        return TypeDecNode(TypeIdentifierNode(id_token.value), type_node)

    def _parse_array_type(self, tokenizer: Tokenizer) -> ArrayTypeNode:
        return ArrayTypeNode()

    def _parse_record_type(self, tokenizer: Tokenizer) -> RecordTypeNode:
        return RecordTypeNode()

    def _parse_func_declaration(self, tokenizer: Tokenizer) -> FuncDecNode:
        _assert_tkn_val(tokenizer.next_token(), Keyword.FUNCTION)

        id_token = tokenizer.next_token()
        _assert_identifier(id_token)

        _assert_tkn_val(tokenizer.next_token(), Punctuation.PAREN_OPEN)
        next_token = tokenizer.next_token()
        field_decs = []
        while next_token.value != Punctuation.PAREN_CLOSE:
            field_decs.append(self._parse_field_declaration(tokenizer))
            next_token = tokenizer.next_token()  # either comma or close paren

        type_id_node = self._get_type_annotation(tokenizer)

        _assert_tkn_val(tokenizer.next_token(), Operator.EQ)
        exp_node = self._parse_expression(tokenizer)
        return FuncDecNode(
            IdentifierNode(id_token.value),
            field_decs,
            exp_node,
            type_identifier=type_id_node,
        )

    def _parse_var_declaration(self, tokenizer: Tokenizer) -> VarDecNode:
        _assert_tkn_val(tokenizer.next_token(), Keyword.VAR)

        id_token = tokenizer.next_token()
        _assert_identifier(id_token)

        type_id = self._get_type_annotation(tokenizer)

        _assert_tkn_val(tokenizer.next_token(), Punctuation.ASSIGNMENT)
        exp_node = self._parse_expression(tokenizer)

        return VarDecNode(
            IdentifierNode(id_token.value),
            exp_node,
            type_identifier=type_id,
        )

    def _parse_field_declaration(self, tokenizer: Tokenizer) -> FieldDecNode:
        id_token = tokenizer.next_token()
        _assert_identifier(id_token)

        type_id = self._get_type_annotation(tokenizer)
        if type_id is None:
            raise ParserException(
                'Field declarations require type annotation.'
            )
        return FieldDecNode(IdentifierNode(id_token.value), type_id)
