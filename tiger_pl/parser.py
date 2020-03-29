import inspect
from abc import ABC
from typing import List, Union, Optional, Callable

from tiger_pl.common import IsClassMixin
from tiger_pl.lexer import (
    Token,
    Keyword,
    TigerLexer,
    Punctuation,
    Operator as OpToken,
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


"""
# Class Heirarchy for AST Nodes
- Every class (except for Python-abstract ones) references a single rule in
  the grammar
- The Python-abstract classes are just a way to organize related rules.
  That is, they're an implementation decision but not required by the language.

AbstractSyntaxTreeNode
    AbstractIdentifier
        Identifier
        TypeIdentifier
    Expression
        IntegerLiteralExpression
        StringLiteralExpression
        LetExpression
        IfThenExpression
        IfThenElseExpression
        WhileExpression
        ForExpression
        BreakExpression
        NilExpNode
        NegationExpression
        SeqExpression
        AbstractLValueExpression
            LValueExpression
            SubscriptExpression
            FieldExpression
        AssignmentExpression
        ArrCreateExpression
        RecCreateExpression
        CallExpression
    FieldCreate
    Declaration
        TypeDeclaration
        VarDeclaration
        FieldDeclaration
        FunDeclaration
    TigerType
    AbstractTigerType
        ArrayType
        RecType
    Operator
"""


class ParserException(Exception):
    pass


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
            "{}={}".format(param.name, repr(getattr(self, param.name, param.default)))
            for param in params.values()
            if param.default != inspect._empty
        ]
        return "{}({}{})".format(
            self.__class__.__name__,
            ", ".join(args),
            ", " + ", ".join(kwargs) if kwargs else "",
        )

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return all(
            getattr(self, p.name, None) == getattr(other, p.name, None)
            for p in inspect.signature(self.__init__).parameters.values()
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

        for param in params[len(args) :]:
            kwarg = kwargs.get(param.name, param.default)
            setattr(self, param.name, kwarg)


class AbstractIdentifier(AbstractSyntaxTreeNode, ABC):
    def __hash__(self):
        return hash(self._value)


class Declaration(AbstractSyntaxTreeNode, IsClassMixin):
    _class_name_base = 'Declaration'


class Expression(AbstractSyntaxTreeNode, IsClassMixin):
    _class_name_base = 'Expression'


class Identifier(AbstractIdentifier):
    def __init__(self, value: str):
        super().__init__(value)


class TypeIdentifier(AbstractIdentifier):
    def __init__(self, value: str):
        super().__init__(value)


class VarDeclaration(Declaration):
    def __init__(
        self,
        identifier: Identifier,
        expression: Expression,
        type_identifier: Optional[TypeIdentifier] = None,
    ):
        super().__init__(identifier, expression, type_identifier=type_identifier)


class FieldDeclaration(Declaration):
    def __init__(self, identifier: Identifier, type_identifier: TypeIdentifier):
        super().__init__(identifier, type_identifier)


class FunDeclaration(Declaration):
    def __init__(
        self,
        identifier: Identifier,
        field_declarations: List[FieldDeclaration],
        expression: Expression,
        type_identifier: Optional[TypeIdentifier] = None,
    ):
        super().__init__(
            identifier, field_declarations, expression, type_identifier=type_identifier
        )


class AbstractTigerType(AbstractSyntaxTreeNode, ABC):
    pass


class ArrayType(AbstractTigerType):
    def __init__(self, type_identifier: TypeIdentifier):
        super().__init__(type_identifier)


class RecType(AbstractTigerType):
    def __init__(self, field_declarations: List[FieldDeclaration]):
        super().__init__(field_declarations)


class TigerType(AbstractSyntaxTreeNode):
    def __init__(self, tiger_type: Union[TypeIdentifier, ArrayType, RecType]):
        super().__init__(tiger_type)


class TypeDeclaration(Declaration):
    def __init__(
        self, type_identifier: TypeIdentifier, tiger_type: TigerType,
    ):
        super().__init__(type_identifier, tiger_type)


class Operator(AbstractSyntaxTreeNode):
    def __init__(self, op: OpToken):
        super().__init__(op)


class IntegerLiteralExpression(Expression):
    def __init__(self, value: int):
        super().__init__(value)


class StringLiteralExpression(Expression):
    def __init__(self, value: str):
        super().__init__(value)


class LetExpression(Expression):
    def __init__(self, declarations: List[Declaration], expressions: List[Expression]):
        if len(declarations) == 0:
            raise ParserException("Let expressions require at least one declaration.")
        super().__init__(declarations, expressions)


class IfThenExpression(Expression):
    def __init__(self, condition: Expression, consequent: Expression):
        super().__init__(condition, consequent)


class IfThenElseExpression(Expression):
    def __init__(
        self, condition: Expression, consequent: Expression, alternative: Expression
    ):
        super().__init__(condition, consequent, alternative)


class WhileExpression(Expression):
    def __ini__(self, condition: Expression, consequent: Expression):
        super().__init__(condition, consequent)


class ForExpression(Expression):
    def __ini__(
        self,
        identfier: Identifier,
        lower_bound: Expression,
        upper_bound: Expression,
        body: Expression,
    ):
        super().__init__(identfier, lower_bound, upper_bound, body)


class BreakExpression(Expression):
    pass


class NilExpression(Expression):
    pass


class NegationExpression(Expression):
    def __init__(self, expression):
        super().__init__(expression)


class SeqExpression(Expression):
    def __init__(self, expressions):
        super().__init__(expressions)


class AbstractLValueExpression(Expression, ABC):
    pass


class LValueExpression(AbstractLValueExpression):
    def __init__(
        self, l_value: Union[Identifier, "SubscriptExpression", "FieldExpression"],
    ):
        super().__init__(l_value)


class SubscriptExpression(AbstractLValueExpression):
    def __init__(
        self, l_value: LValueExpression, index_expression: Expression,
    ):
        super().__init__(l_value, index_expression)


class FieldExpression(AbstractLValueExpression):
    def __init__(
        self, l_value: LValueExpression, identifer: Identifier,
    ):
        super().__init__(l_value, identifer)


class AssignmentExpression(Expression):
    def __init__(
        self, l_value: LValueExpression, expression: Expression,
    ):
        super().__init__(l_value, expression)


class ArrCreateExpression(Expression):
    def __init__(
        self,
        type_identifier: Identifier,
        size_expression: Expression,
        initial_values_expression: Expression,
    ):
        super().__init__(type_identifier, size_expression, initial_values_expression)


class FieldCreate(AbstractSyntaxTreeNode):
    def __init__(
        self, identifier: Identifier, expression: Expression,
    ):
        super().__init__(identifier, expression)


class RecCreateExpression(Expression):
    def __init__(
        self, type_identifier: Identifier, field_create_expressions: List[FieldCreate],
    ):
        super().__init__(type_identifier, field_create_expressions)


class CallExpression(Expression):
    def __init__(
        self, identifier: Identifier, expressions: List[Expression],
    ):
        super().__init__(identifier, expressions)


class InfixExpression(Expression):
    def __init__(
        self,
        left_expression: Expression,
        operator: Operator,
        right_expression: Expression,
    ):
        super().__init__(left_expression, operator, right_expression)


class Program(AbstractSyntaxTreeNode):
    """
    For use as the root node of the AST
    """

    def __init__(self, expression: Expression):
        super().__init__(expression)


def _assert_tkn_val(token: Token, expect_val: Union[str, List[str]]):
    if isinstance(expect_val, list):
        if token not in expect_val:
            raise ParserException(
                'Expected one of {}, found "{}"'.format(expect_val, token.value)
            )
    else:
        if token != expect_val:
            raise ParserException(
                'Expected "{}", found "{}"'.format(expect_val, token.value)
            )


def _assert_identifier(token: Token):
    if not token.is_identifier:
        raise ParserException("Expected identifier, found {}".format(token))


class TigerParser(object):
    @staticmethod
    def _get_type_annotation(lexer: TigerLexer) -> Optional[TypeIdentifier]:
        """
        Matches the common, optional type annotation pattern => ': tyId'
        """
        if lexer.peek() == Punctuation.COLON:
            lexer.next()
            type_id_token = lexer.next()
            _assert_identifier(type_id_token)
            return TypeIdentifier(type_id_token.value)
        else:
            return None

    @staticmethod
    def _collect_list(
        lexer: TigerLexer,
        list_item_parser: Callable[[TigerLexer], AbstractSyntaxTreeNode],
        termination: Token,
        separator: Optional[Token] = None,
    ) -> List[AbstractSyntaxTreeNode]:
        """
        Matches the common pattern of a list of items enclosed by some token.
        Often list items are separated by a token as well.

        E.g. ( exp*, )
        """
        items = []
        while True:
            if lexer.peek() == termination:
                lexer.next()
                break
            items.append(list_item_parser(lexer))
            if separator and lexer.peek() == separator:
                lexer.next()
        return items

    def parse(self, lexer: TigerLexer) -> Program:
        next_token = lexer.peek()
        if next_token == Keyword.LET:
            return Program(self._parse_let_expression(lexer))
        else:
            # TODO: support programs not wrapped in a let
            raise NotImplementedError()

    def _parse_expression(self, lexer: TigerLexer) -> Expression:
        """
        exp → lValue | nil | intLit | stringLit
            | seqExp | negation | callExp | infixExp
            | arrCreate | recCreate | assignment
            | ifThenElse | ifThen | whileExp | forExp
            | break | letExp
        """
        next_token = lexer.peek()
        if next_token.is_integer_literal:
            exp = IntegerLiteralExpression(lexer.next().value)
        elif next_token.is_string_literal:
            exp = StringLiteralExpression(lexer.next().value)
        elif next_token == Keyword.NIL:
            exp = self._parse_nil_expression(lexer)
        elif next_token == Keyword.BREAK:
            exp = self._parse_break_expression(lexer)
        elif next_token == Keyword.IF:
            exp = self._parse_if_expression(lexer)
        elif next_token == Keyword.WHILE:
            exp = self._parse_while_expression(lexer)
        elif next_token == Keyword.FOR:
            exp = self._parse_for_expression(lexer)
        elif next_token == Keyword.LET:
            exp = self._parse_let_expression(lexer)
        elif next_token == OpToken.SUB:
            exp = self._parse_negation_expression(lexer)
        elif next_token == Punctuation.PAREN_OPEN:
            exp = self._parse_seq_expression(lexer)
        elif next_token.is_identifier:
            next_next = lexer.peek(1)
            if next_next == Punctuation.DOT:
                exp = self._parse_new_field_expression(lexer)
            elif next_next == Punctuation.BRACKET_OPEN:
                exp = self._parse_arr_create_or_subscript_expression(lexer)
            elif next_next == Punctuation.PAREN_OPEN:
                exp = self._parse_call_expression(lexer)
            elif next_next == Punctuation.CURLY_OPEN:
                exp = self._parse_rec_create_expression(lexer)
            else:
                raise ParserException("Unexpected token {}".format(next_token.value))
        else:
            raise ParserException("Unexpected token {}".format(next_token.value))

        if lexer.peek() in OpToken.values():
            op_token = lexer.next()
            right_exp = self._parse_expression(lexer)
            return InfixExpression(exp, Operator(op_token), right_exp,)
        else:
            return exp

    def _parse_let_expression(self, lexer: TigerLexer,) -> LetExpression:
        """
        letExp → let dec+ in exp∗; end
        """
        _assert_tkn_val(lexer.next(), Keyword.LET)
        next_token = lexer.peek()
        decs = []
        while next_token in [Keyword.TYPE, Keyword.FUNCTION, Keyword.VAR]:
            if next_token == Keyword.TYPE:
                dec_node = self._parse_type_declaration(lexer)
            elif next_token == Keyword.FUNCTION:
                dec_node = self._parse_func_declaration(lexer)
            else:
                dec_node = self._parse_var_declaration(lexer)
            decs.append(dec_node)
            next_token = lexer.next()

        _assert_tkn_val(next_token, Keyword.IN)

        exps = self._collect_list(lexer, self._parse_expression, Keyword.END,)

        return LetExpression(decs, exps)

    def _parse_if_expression(
        self, lexer: TigerLexer
    ) -> Union[IfThenExpression, IfThenElseExpression]:
        """
        ifthenelse → if exp then exp else exp
        ifthen → if exp then exp
        """
        _assert_tkn_val(lexer.next(), Keyword.IF)
        cond_exp = self._parse_expression(lexer)
        _assert_tkn_val(lexer.next(), Keyword.THEN)
        body_exp = self._parse_expression(lexer)
        next_token = lexer.next()
        if next_token == Keyword.ELSE:
            else_body_exp = self._parse_expression(lexer)
            return IfThenElseExpression(cond_exp, body_exp, else_body_exp)
        else:
            return IfThenExpression(cond_exp, body_exp)

    def _parse_while_expression(self, lexer: TigerLexer) -> WhileExpression:
        """
        whileExp → while exp do exp
        """
        _assert_tkn_val(lexer.next(), Keyword.WHILE)
        cond_exp = self._parse_expression(lexer)
        _assert_tkn_val(lexer.next(), Keyword.DO)
        body_exp = self._parse_expression(lexer)
        return WhileExpression(cond_exp, body_exp)

    def _parse_for_expression(self, lexer: TigerLexer) -> ForExpression:
        """
        forExp → for id := exp to exp do exp
        """
        _assert_tkn_val(lexer.next(), Keyword.FOR)
        id_token = lexer.next()
        _assert_identifier(id_token)
        _assert_tkn_val(lexer.next(), Keyword.ASSIGMENT)
        lower_bound = self._parse_expression(lexer)
        _assert_tkn_val(lexer.next(), Keyword.TO)
        upper_bound = self._parse_expression(lexer)
        _assert_tkn_val(lexer.next(), Keyword.DO)
        body = self._parse_expression(lexer)
        return ForExpression(
            Identifier(id_token.value), lower_bound, upper_bound, body,
        )

    def _parse_break_expression(self, lexer: TigerLexer) -> BreakExpression:
        """
        break
        """
        _assert_tkn_val(lexer.next(), Keyword.BREAK)
        return BreakExpression()

    def _parse_nil_expression(self, lexer: TigerLexer) -> NilExpression:
        """
        nil
        """
        _assert_tkn_val(lexer.next(), Keyword.NIL)
        return NilExpression()

    def _parse_negation_expression(self, lexer: TigerLexer) -> NegationExpression:
        """
        negation → - exp
        """
        _assert_tkn_val(lexer.next(), OpToken.SUB)
        exp = self._parse_expression(lexer)
        return NegationExpression(exp)

    def _parse_seq_expression(self, lexer: TigerLexer) -> SeqExpression:
        """
        seqExp → ( exp∗; )
        """
        _assert_tkn_val(lexer.next(), Punctuation.PAREN_OPEN)
        exps = []
        next_token = lexer.next()
        while next_token != Punctuation.PAREN_CLOSE:
            if exps:
                _assert_tkn_val(lexer.next(), Punctuation.SEMI_COLON)
            exps.append(self._parse_expression(lexer))
            next_token = lexer.next()
        return SeqExpression(exps)

    def _parse_arr_create_or_subscript_expression(
        self, lexer: TigerLexer
    ) -> Union[LValueExpression, ArrCreateExpression, AssignmentExpression]:
        """
        arrCreate → tyId [ exp ] of exp
        subscript → lValue [ exp ]
        """
        id_token = lexer.next()
        _assert_identifier(id_token)

        _assert_tkn_val(lexer.next(), Punctuation.BRACKET_OPEN)
        exp = self._parse_expression(lexer)
        _assert_tkn_val(lexer.next(), Punctuation.BRACKET_CLOSE)

        if lexer.peek() == Keyword.OF:
            lexer.next()
            of_exp = self._parse_expression(lexer)
            return ArrCreateExpression(Identifier(id_token.value), exp, of_exp)

        subscript_exp = LValueExpression(
            SubscriptExpression(LValueExpression(Identifier(id_token.value)), exp,)
        )
        return self._parse_rest_of_l_value_expression(subscript_exp, lexer)

    def _parse_new_field_expression(
        self, lexer: TigerLexer
    ) -> Union[LValueExpression, AssignmentExpression]:
        """
        fieldExp → lValue . id
        """
        left_id_token = lexer.next()
        _assert_identifier(left_id_token)
        _assert_tkn_val(lexer.next(), Punctuation.DOT)
        right_id_token = lexer.next()
        _assert_identifier(right_id_token)
        field_exp = LValueExpression(
            FieldExpression(
                LValueExpression(Identifier(left_id_token.value)),
                Identifier(right_id_token.value),
            )
        )
        return self._parse_rest_of_l_value_expression(field_exp, lexer)

    def _parse_rest_of_l_value_expression(
        self, l_value: LValueExpression, lexer: TigerLexer
    ) -> Union[LValueExpression, AssignmentExpression]:
        """
        lValue → id | subscript | fieldExp
        subscript → lValue [ exp ]
        fieldExp → lValue . id

        The base case for l-values is an identifier, which is also the
        left-most token in the expression. This makes it pretty pretty easy to
        spot a new l-value starting (`a.`, `a[`), but then it may nest
        indefinitely (e.g. `a.b[c].d`).

        The calling method is responsible for the left-most l-value. This
        method is for finishing it off.
        """
        while lexer.peek() in [Punctuation.BRACKET_OPEN, Punctuation.DOT]:
            next_token = lexer.next()
            if next_token == Punctuation.BRACKET_OPEN:
                exp = self._parse_expression(lexer)
                l_value = LValueExpression(SubscriptExpression(l_value, exp))
                _assert_tkn_val(lexer.next(), Punctuation.BRACKET_CLOSE)
            elif next_token == Punctuation.DOT:
                id_token = lexer.next()
                l_value = LValueExpression(
                    FieldExpression(l_value, Identifier(id_token.value))
                )

        if lexer.peek() == Punctuation.ASSIGNMENT:
            return self._parse_assignment_expression(l_value, lexer)
        else:
            return l_value

    def _parse_assignment_expression(
        self, l_value: LValueExpression, lexer: TigerLexer,
    ) -> AssignmentExpression:
        """
        assignment → lValue := exp
        """
        _assert_tkn_val(lexer.next(), Punctuation.ASSIGNMENT)
        exp = self._parse_expression(lexer)
        return AssignmentExpression(l_value, exp)

    def _parse_call_expression(self, lexer: TigerLexer,) -> CallExpression:
        """
        callExp → id ( exp∗, )
        """
        id_token = lexer.next()
        _assert_identifier(id_token)
        _assert_tkn_val(lexer.next(), Punctuation.PAREN_OPEN)
        exps = self._collect_list(
            lexer,
            self._parse_expression,
            Punctuation.PAREN_CLOSE,
            separator=Punctuation.COMMA,
        )

        return CallExpression(Identifier(id_token.value), exps)

    def _parse_rec_create_expression(self, lexer: TigerLexer,) -> RecCreateExpression:
        """
        recCreate → tyId { fieldCreate∗, }
        """
        id_token = lexer.next()
        _assert_identifier(id_token)
        _assert_tkn_val(lexer.next(), Punctuation.CURLY_OPEN)
        field_creates = self._collect_list(
            lexer,
            self._parse_field_create,
            Punctuation.CURLY_CLOSE,
            separator=Punctuation.COMMA,
        )
        return RecCreateExpression(Identifier(id_token.value), field_creates,)

    def _parse_field_create(self, lexer: TigerLexer) -> FieldCreate:
        id_token = lexer.next()
        _assert_tkn_val(lexer.next(), OpToken.EQ)
        exp = self._parse_expression(lexer)
        return FieldCreate(Identifier(id_token.value), exp)

    def _parse_type_declaration(self, lexer: TigerLexer) -> TypeDeclaration:
        """
        tyDec → type tyId = ty
        """
        _assert_tkn_val(lexer.next(), Keyword.TYPE)

        id_token = lexer.next()
        _assert_identifier(id_token)

        _assert_tkn_val(lexer.next(), OpToken.EQ)

        next_token = lexer.peek()
        if next_token.is_identifier:
            type_node = TypeIdentifier(next_token.value)
        elif next_token == Keyword.ARRAY:
            type_node = self._parse_array_type(lexer)
        elif next_token == Punctuation.CURLY_OPEN:
            type_node = self._parse_record_type(lexer)
        else:
            raise ParserException(
                "Expected type identifier, array type, or record type. "
                'Found "{}"'.format(next_token)
            )
        return TypeDeclaration(TypeIdentifier(id_token.value), type_node)

    def _parse_array_type(self, lexer: TigerLexer) -> ArrayType:
        """
        arrTy → array of tyId
        """
        _assert_tkn_val(lexer.next(), Keyword.ARRAY)
        _assert_tkn_val(lexer.next(), Keyword.OF)
        type_id_token = lexer.next()
        return ArrayType(TypeIdentifier(type_id_token.value))

    def _parse_record_type(self, lexer: TigerLexer) -> RecType:
        """
        recTy → { fieldDec∗, }
        """
        _assert_tkn_val(lexer.next(), Punctuation.CURLY_OPEN)
        field_decs = self._collect_list(
            lexer,
            self._parse_field_declaration,
            Punctuation.CURLY_CLOSE,
            separator=Punctuation.COMMA,
        )
        return RecType(field_decs)

    def _parse_func_declaration(self, lexer: TigerLexer) -> FunDeclaration:
        """
        funDec → function id ( fieldDec∗, ) = exp
                | function id ( fieldDec∗, ) : tyId = exp
        """
        _assert_tkn_val(lexer.next(), Keyword.FUNCTION)

        id_token = lexer.next()
        _assert_identifier(id_token)

        _assert_tkn_val(lexer.next(), Punctuation.PAREN_OPEN)
        next_token = lexer.next()
        field_decs = []
        while next_token != Punctuation.PAREN_CLOSE:
            field_decs.append(self._parse_field_declaration(lexer))
            next_token = lexer.next()  # either comma or close paren

        type_id_node = self._get_type_annotation(lexer)

        _assert_tkn_val(lexer.next(), OpToken.EQ)
        exp_node = self._parse_expression(lexer)
        return FunDeclaration(
            Identifier(id_token.value),
            field_decs,
            exp_node,
            type_identifier=type_id_node,
        )

    def _parse_var_declaration(self, lexer: TigerLexer,) -> VarDeclaration:
        """
        varDec → var id := exn
                | var id : tyId := exp
        """
        _assert_tkn_val(lexer.next(), Keyword.VAR)

        id_token = lexer.next()
        _assert_identifier(id_token)

        type_id = self._get_type_annotation(lexer)

        _assert_tkn_val(lexer.next(), Punctuation.ASSIGNMENT)
        exp_node = self._parse_expression(lexer)

        return VarDeclaration(
            Identifier(id_token.value), exp_node, type_identifier=type_id,
        )

    def _parse_field_declaration(self, lexer: TigerLexer) -> FieldDeclaration:
        """
        fieldDec → id : tyId
        """
        id_token = lexer.next()
        _assert_identifier(id_token)

        type_id = self._get_type_annotation(lexer)
        if type_id is None:
            raise ParserException("Field declarations require type annotation.")
        return FieldDeclaration(Identifier(id_token.value), type_id)
