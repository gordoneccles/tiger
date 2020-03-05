import inspect
from abc import ABC
from typing import List, Union, Optional

from tiger_interpreter.tokenizer import (
    Token, Keyword, TigerTokenizer, Punctuation, Operator,
)

"""
# Grammar Reference

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
ifthenelse → if exp then exp else exp
ifthen → if exp then exp
whileExp → while exp do exp
forExp → for id := exp to exp do exp
letExp → let dec+ in exp∗; end
"""


class ParserException(Exception):
    pass


"""
# Class Heirarchy for AST Nodes
AbstractSyntaxTreeNode
    IdentifierNode
    TypeIdentifierNode
    Expression
        LetExpNode
        NilExpNode
        IntegerLiteralNode
        StringLiteralNode
        (...more to come)
    Declaration
        TypeDecNode
        VarDecNode
        FieldDecNode
        FuncDecNode
    TigerType
        ArrayType
        RecordType
"""


class AbstractSyntaxTreeNode(ABC):
    """
    Both "abstract" in the syntax tree sense and the python sense :)
    """

    def __repr__(self):
        """
        Visualizes instances of this class like
        ClassName(this_arg, that_kwarg='some_default')
        """
        params = inspect.signature(self.__init__).parameters
        args = [
            repr(getattr(self, param.name))
            for param in params.values()
            if param.default == inspect._empty
        ]
        kwargs = [
            '{}={}'.format(
                param.name,
                repr(getattr(self, param.name, param.default))
            )
            for param in params.values()
            if param.default != inspect._empty
        ]
        return '{}({}{})'.format(
            self.__class__.__name__,
            ', '.join(args),
            ', ' + ', '.join(kwargs) if kwargs else '',
        )

    def __init__(self, *args, **kwargs):
        """
        A generic equivalent of
        def __init__(self, arg1, arg2=True):
            self.arg1 = arg1
            self.arg2 = arg2

        Basically saves some boilerplate in subclasses by using super().
        Also enforces the init signature as the source of truth.
        """
        params = list(inspect.signature(self.__init__).parameters.values())
        for arg, param in zip(args, params):
            setattr(self, param.name, arg)

        for param in params[len(args):]:
            kwarg = kwargs.get(param.name, param.default)
            setattr(self, param.name, kwarg)


class IdentifierNode(AbstractSyntaxTreeNode):

    def __init__(self, value: str):
        super().__init__(value)


class TypeIdentifierNode(AbstractSyntaxTreeNode):

    def __init__(self, value: str):
        super().__init__(value)


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
        super().__init__(declarations, expressions)


class IfThenExpression(Expression):
    def __init__(self, condition: Expression, consequent: Expression):
        super().__init__(condition, consequent)


class IfThenElseExpression(Expression):
    def __init__(
        self,
        condition: Expression,
        consequent: Expression,
        alternative: Expression
    ):
        super().__init__(condition, consequent, alternative)


class NilExpNode(Expression):
    pass


class IntegerLiteralExpNode(Expression):

    def __init__(self, value: int):
        super().__init__(value)


class StringLiteralExpNode(Expression):

    def __init__(self, value: str):
        super().__init__(value)


class TigerType(AbstractSyntaxTreeNode):
    pass


class ArrayTypeNode(TigerType):
    # TODO
    pass


class RecordTypeNode(TigerType):
    # TODO
    pass


class TypeDecNode(Declaration):
    def __init__(
        self,
        type_identifier: TypeIdentifierNode,
        tiger_type: TigerType,
    ):
        super().__init__(type_identifier, tiger_type)


class VarDecNode(Declaration):

    def __init__(
        self,
        identifier: IdentifierNode,
        expression: Expression,
        type_identifier: Optional[TypeIdentifierNode] = None,
    ):
        super().__init__(
            identifier, expression, type_identifier=type_identifier
        )


class FieldDecNode(Declaration):
    def __init__(
        self, identifier: IdentifierNode, type_identifier: TypeIdentifierNode
    ):
        super().__init__(identifier, type_identifier)


class FuncDecNode(Declaration):
    def __init__(
        self,
        identifier: IdentifierNode,
        field_declarations: List[FieldDecNode],
        expression: Expression,
        type_identifier: Optional[TypeIdentifierNode] = None,
    ):
        super().__init__(
            identifier,
            field_declarations,
            expression,
            type_identifier=type_identifier
        )


def _assert_tkn_val(token: Token, expect_val: Union[str, List[str]]):
    if isinstance(expect_val, list):
        if token not in expect_val:
            raise ParserException(
                'Expected one of {}, found "{}"'.format(
                    expect_val, token.value
                )
            )
    else:
        if token != expect_val:
            raise ParserException(
                'Expected "{}", found "{}"'.format(expect_val, token.value)
            )


def _assert_identifier(token: Token):
    if not token.is_identifier:
        raise ParserException('Expected identifier, found {}'.format(token))


class Program(AbstractSyntaxTreeNode):
    """
    For use as the root node of the AST
    """
    def __init__(self, expression: Expression):
        super().__init__(expression)


class TigerParser(object):

    @staticmethod
    def _get_type_annotation(
        tokenizer: TigerTokenizer
    ) -> Optional[TypeIdentifierNode]:
        """
        Matches the common, optional type annotation pattern => ': tyId'
        """
        if tokenizer.next_token() == Punctuation.COLON:
            type_id_token = tokenizer.next_token()
            _assert_identifier(type_id_token)
            return TypeIdentifierNode(type_id_token)
        else:
            tokenizer.rewind()
            return None

    def parse(self, tokenizer: TigerTokenizer) -> Program:
        next_token = tokenizer.next_token()
        tokenizer.rewind()
        if next_token == Keyword.LET:
            return Program(self._parse_let_expression(tokenizer))
        else:
            # TODO: support programs not wrapped in a let (ie. seq programs)
            raise NotImplementedError()

    def _parse_expression(self, tokenizer: TigerTokenizer) -> Expression:
        """
        exp → lValue | nil | intLit | stringLit
            | seqExp | negation | callExp | infixExp
            | arrCreate | recCreate | assignment
            | ifThenElse | ifThen | whileExp | forExp
            | break | letExp
        """
        next_token = tokenizer.next_token()
        if next_token == Keyword.NIL:
            return NilExpNode()
        elif next_token.is_integer_literal:
            return IntegerLiteralExpNode(next_token.value)
        elif next_token.is_string_literal:
            return StringLiteralExpNode(next_token.value)
        elif next_token == Keyword.IF:
            tokenizer.rewind()
            return self._parse_if_expression(tokenizer)
        else:
            raise NotImplementedError('TODO')

    def _parse_let_expression(self, tokenizer: TigerTokenizer) -> LetExpNode:
        """
        letExp → let dec+ in exp∗; end
        """
        _assert_tkn_val(tokenizer.next_token(), Keyword.LET)
        next_token = tokenizer.next_token()
        tokenizer.rewind()
        decs = []
        while next_token in [Keyword.TYPE, Keyword.FUNCTION, Keyword.VAR]:
            if next_token == Keyword.TYPE:
                dec_node = self._parse_type_declaration(tokenizer)
            elif next_token == Keyword.FUNCTION:
                dec_node = self._parse_func_declaration(tokenizer)
            else:
                dec_node = self._parse_var_declaration(tokenizer)
            decs.append(dec_node)
            next_token = tokenizer.next_token()

        _assert_tkn_val(tokenizer.next_token(), Keyword.IN)

        next_token = tokenizer.next_token()
        exps = []
        while next_token != Keyword.END:
            exps.append(self._parse_expression(tokenizer))
            next_token = tokenizer.next_token()
            if next_token == Punctuation.SEMI_COLON:
                next_token = tokenizer.next_token()

        return LetExpNode(decs, exps)

    def _parse_if_expression(
        self, tokenizer: TigerTokenizer
    ) -> Union[IfThenExpression, IfThenElseExpression]:
        """
        ifthenelse → if exp then exp else exp
        ifthen → if exp then exp
        """
        _assert_tkn_val(tokenizer.next_token(), Keyword.IF)
        cond_exp = self._parse_expression(tokenizer)
        _assert_tkn_val(tokenizer.next_token(), Keyword.THEN)
        body_exp = self._parse_expression(tokenizer)
        next_token = tokenizer.next_token()
        if next_token == Keyword.ELSE:
            else_body_exp = self._parse_expression(tokenizer)
            return IfThenElseExpression(
                cond_exp, body_exp, else_body_exp
            )
        else:
            return IfThenExpression(cond_exp, body_exp)

    def _parse_type_declaration(
        self, tokenizer: TigerTokenizer
    ) -> TypeDecNode:
        """
        tyDec → type tyId = ty
        """
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
                'Found "{}"'.format(next_token)
            )
        return TypeDecNode(TypeIdentifierNode(id_token.value), type_node)

    def _parse_array_type(self, tokenizer: TigerTokenizer) -> ArrayTypeNode:
        """
        arrTy → array of tyId
        """
        return ArrayTypeNode()

    def _parse_record_type(self, tokenizer: TigerTokenizer) -> RecordTypeNode:
        """
        recTy → { fieldDec∗, }
        """
        return RecordTypeNode()

    def _parse_func_declaration(
        self, tokenizer: TigerTokenizer
    ) -> FuncDecNode:
        """
        funDec → function id ( fieldDec∗, ) = exp
                | function id ( fieldDec∗, ) : tyId = exp
        """
        _assert_tkn_val(tokenizer.next_token(), Keyword.FUNCTION)

        id_token = tokenizer.next_token()
        _assert_identifier(id_token)

        _assert_tkn_val(tokenizer.next_token(), Punctuation.PAREN_OPEN)
        next_token = tokenizer.next_token()
        field_decs = []
        while next_token != Punctuation.PAREN_CLOSE:
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

    def _parse_var_declaration(self, tokenizer: TigerTokenizer) -> VarDecNode:
        """
        varDec → var id := exn
                | var id : tyId := exp
        """
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

    def _parse_field_declaration(
        self, tokenizer: TigerTokenizer
    ) -> FieldDecNode:
        """
        fieldDec → id : tyId
        """
        id_token = tokenizer.next_token()
        _assert_identifier(id_token)

        type_id = self._get_type_annotation(tokenizer)
        if type_id is None:
            raise ParserException(
                'Field declarations require type annotation.'
            )
        return FieldDecNode(IdentifierNode(id_token.value), type_id)
