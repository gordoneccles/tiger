import inspect
from abc import ABC
from typing import List, Union, Optional

from tiger_interpreter.lexer import (
    Token, Keyword, TigerLexer, Punctuation, Operator as OpToken,
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


class AbstractIdentifier(AbstractSyntaxTreeNode, ABC):
    pass


class Declaration(AbstractSyntaxTreeNode):
    pass


class Expression(AbstractSyntaxTreeNode):
    pass


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
        super().__init__(
            identifier, expression, type_identifier=type_identifier
        )


class FieldDeclaration(Declaration):
    def __init__(
        self, identifier: Identifier, type_identifier: TypeIdentifier
    ):
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
            identifier,
            field_declarations,
            expression,
            type_identifier=type_identifier
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
    def __init__(
        self,
        tiger_type: Union[TypeIdentifier, ArrayType, RecType]
    ):
        super().__init__(tiger_type)


class TypeDeclaration(Declaration):
    def __init__(
        self,
        type_identifier: TypeIdentifier,
        tiger_type: TigerType,
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


class WhileExpression(Expression):
    def __ini__(
        self,
        condition: Expression,
        consequent: Expression,
    ):
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
        self,
        l_value: Union[
            Identifier, 'SubscriptExpression', 'FieldExpression'
        ],
    ):
        super().__init__(l_value)


class SubscriptExpression(AbstractLValueExpression):
    def __init__(
        self,
        l_value: LValueExpression,
        index_expression: Expression,
    ):
        super().__init__(l_value, index_expression)


class FieldExpression(AbstractLValueExpression):
    def __init__(
        self,
        l_value: LValueExpression,
        identifer: Identifier,
    ):
        super().__init__(l_value, identifer)


class AssignmentExpression(Expression):
    def __init__(
        self,
        l_value: LValueExpression,
        expression: Expression,
    ):
        super().__init__(l_value, expression)


class ArrCreateExpression(Expression):
    def __init__(
        self,
        type_identifier: Identifier,
        size_expression: Expression,
        initial_values_expression: Expression,
    ):
        super().__init__(
            type_identifier, size_expression, initial_values_expression
        )


class FieldCreate(AbstractSyntaxTreeNode):
    def __init__(
        self,
        identifier: Identifier,
        expression: Expression,
    ):
        super().__init__(identifier, expression)


class RecCreateExpression(Expression):
    def __init__(
        self,
        type_identifier: Identifier,
        field_create_expressions: List[FieldCreate],
    ):
        super().__init__(type_identifier, field_create_expressions)


class CallExpression(Expression):
    def __init__(
        self,
        identifier: Identifier,
        expressions: List[Expression],
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


class ParserException(Exception):
    pass


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


class TigerParser(object):

    @staticmethod
    def _get_type_annotation(
        tokenizer: TigerLexer
    ) -> Optional[TypeIdentifier]:
        """
        Matches the common, optional type annotation pattern => ': tyId'
        """
        if tokenizer.peek() == Punctuation.COLON:
            tokenizer.next()
            type_id_token = tokenizer.next()
            _assert_identifier(type_id_token)
            return TypeIdentifier(type_id_token)
        else:
            return None

    def parse(self, tokenizer: TigerLexer) -> Program:
        next_token = tokenizer.peek()
        if next_token == Keyword.LET:
            return Program(self._parse_let_expression(tokenizer))
        else:
            # TODO: support programs not wrapped in a let (ie. seq programs)
            raise NotImplementedError()

    def _parse_expression(self, tokenizer: TigerLexer) -> Expression:
        """
        exp → lValue | nil | intLit | stringLit
            | seqExp | negation | callExp | infixExp
            | arrCreate | recCreate | assignment
            | ifThenElse | ifThen | whileExp | forExp
            | break | letExp

        TODO:
            infixExp → exp infixOp exp
        """
        next_token = tokenizer.peek()
        if next_token.is_integer_literal:
            exp = IntegerLiteralExpression(tokenizer.next().value)
        elif next_token.is_string_literal:
            exp = StringLiteralExpression(tokenizer.next().value)
        elif next_token == Keyword.NIL:
            exp = self._parse_nil_expression(tokenizer)
        elif next_token == Keyword.BREAK:
            exp = self._parse_break_expression(tokenizer)
        elif next_token == Keyword.IF:
            exp = self._parse_if_expression(tokenizer)
        elif next_token == Keyword.WHILE:
            exp = self._parse_while_expression(tokenizer)
        elif next_token == Keyword.FOR:
            exp = self._parse_for_expression(tokenizer)
        elif next_token == Keyword.LET:
            exp = self._parse_let_expression(tokenizer)
        elif next_token == OpToken.SUB:
            exp = self._parse_negation_expression(tokenizer)
        elif next_token == Punctuation.PAREN_OPEN:
            exp = self._parse_seq_expression(tokenizer)
        elif next_token.is_identifier:
            next_next = tokenizer.peek(1)
            if next_next == Punctuation.DOT:
                exp = self._parse_new_field_expression(tokenizer)
            elif next_next == Punctuation.BRACKET_OPEN:
                exp = self._parse_arr_create_or_subscript_expression(
                    tokenizer
                )
            elif next_next == Punctuation.PAREN_OPEN:
                exp = self._parse_call_expression(tokenizer)
            elif next_next == Punctuation.CURLY_OPEN:
                exp = self._parse_rec_create_expression(tokenizer)
            else:
                raise ParserException(
                    'Unexpected token {}'.format(next_token.value)
                )
        else:
            raise ParserException(
                'Unexpected token {}'.format(next_token.value)
            )
        
        if tokenizer.peek() in OpToken.values:
            op_token = tokenizer.next()
            right_exp = self._parse_expression(tokenizer)
            return InfixExpression(
                exp,
                OpToken()
            )
            

    def _parse_let_expression(
        self,
        tokenizer: TigerLexer,
    ) -> LetExpression:
        """
        letExp → let dec+ in exp∗; end
        """
        _assert_tkn_val(tokenizer.next(), Keyword.LET)
        next_token = tokenizer.peek()
        decs = []
        while next_token in [Keyword.TYPE, Keyword.FUNCTION, Keyword.VAR]:
            if next_token == Keyword.TYPE:
                dec_node = self._parse_type_declaration(tokenizer)
            elif next_token == Keyword.FUNCTION:
                dec_node = self._parse_func_declaration(tokenizer)
            else:
                dec_node = self._parse_var_declaration(tokenizer)
            decs.append(dec_node)
            next_token = tokenizer.next()

        _assert_tkn_val(tokenizer.next(), Keyword.IN)

        next_token = tokenizer.next()
        exps = []
        while next_token != Keyword.END:
            exps.append(self._parse_expression(tokenizer))
            next_token = tokenizer.next()
            if next_token == Punctuation.SEMI_COLON:
                next_token = tokenizer.next()

        return LetExpression(decs, exps)

    def _parse_if_expression(
        self, tokenizer: TigerLexer
    ) -> Union[IfThenExpression, IfThenElseExpression]:
        """
        ifthenelse → if exp then exp else exp
        ifthen → if exp then exp
        """
        _assert_tkn_val(tokenizer.next(), Keyword.IF)
        cond_exp = self._parse_expression(tokenizer)
        _assert_tkn_val(tokenizer.next(), Keyword.THEN)
        body_exp = self._parse_expression(tokenizer)
        next_token = tokenizer.next()
        if next_token == Keyword.ELSE:
            else_body_exp = self._parse_expression(tokenizer)
            return IfThenElseExpression(cond_exp, body_exp, else_body_exp)
        else:
            return IfThenExpression(cond_exp, body_exp)

    def _parse_while_expression(
        self, tokenizer: TigerLexer
    ) -> WhileExpression:
        """
        whileExp → while exp do exp
        """
        _assert_tkn_val(tokenizer.next(), Keyword.WHILE)
        cond_exp = self._parse_expression(tokenizer)
        _assert_tkn_val(tokenizer.next(), Keyword.DO)
        body_exp = self._parse_expression(tokenizer)
        return WhileExpression(cond_exp, body_exp)

    def _parse_for_expression(
        self, tokenizer: TigerLexer
    ) -> ForExpression:
        """
        forExp → for id := exp to exp do exp
        """
        _assert_tkn_val(tokenizer.next(), Keyword.FOR)
        id_token = tokenizer.next()
        _assert_identifier(id_token)
        _assert_tkn_val(tokenizer.next(), Keyword.ASSIGMENT)
        lower_bound = self._parse_expression(tokenizer)
        _assert_tkn_val(tokenizer.next(), Keyword.TO)
        upper_bound = self._parse_expression(tokenizer)
        _assert_tkn_val(tokenizer.next(), Keyword.DO)
        body = self._parse_expression(tokenizer)
        return ForExpression(
            Identifier(id_token.value),
            lower_bound,
            upper_bound,
            body,
        )

    def _parse_break_expression(
        self, tokenizer: TigerLexer
    ) -> BreakExpression:
        """
        break
        """
        _assert_tkn_val(tokenizer.next(), Keyword.BREAK)
        return BreakExpression()

    def _parse_nil_expression(
        self, tokenizer: TigerLexer
    ) -> NilExpression:
        """
        nil
        """
        _assert_tkn_val(tokenizer.next(), Keyword.NIL)
        return NilExpression()

    def _parse_negation_expression(
        self, tokenizer: TigerLexer
    ) -> NegationExpression:
        """
        negation → - exp
        """
        _assert_tkn_val(tokenizer.next(), OpToken.SUB)
        exp = self._parse_expression(tokenizer)
        return NegationExpression(exp)

    def _parse_seq_expression(
        self, tokenizer: TigerLexer
    ) -> SeqExpression:
        """
        seqExp → ( exp∗; )
        """
        _assert_tkn_val(tokenizer.next(), Punctuation.PAREN_OPEN)
        exps = []
        next_token = tokenizer.next()
        while next_token != Punctuation.PAREN_CLOSE:
            if exps:
                _assert_tkn_val(tokenizer.next(), Punctuation.SEMI_COLON)
            exps.append(self._parse_expression(tokenizer))
            next_token = tokenizer.next()
        return SeqExpression(exps)

    def _parse_arr_create_or_subscript_expression(
        self, tokenizer: TigerLexer
    ) -> Union[LValueExpression, ArrCreateExpression, AssignmentExpression]:
        """
        arrCreate → tyId [ exp ] of exp
        subscript → lValue [ exp ]
        """
        id_token = tokenizer.next()
        _assert_identifier(id_token)

        _assert_tkn_val(tokenizer.next(), Punctuation.BRACKET_OPEN)
        exp = self._parse_expression(tokenizer)
        _assert_tkn_val(tokenizer.next(), Punctuation.BRACKET_CLOSE)
        
        if tokenizer.peek() == Keyword.OF:
            tokenizer.next()
            of_exp = self._parse_expression(tokenizer)
            return ArrCreateExpression(
                Identifier(id_token.value),
                exp,
                of_exp
            )

        subscript_exp = LValueExpression(
            SubscriptExpression(
                LValueExpression(Identifier(id_token.value)),
                exp,
            )
        )
        return self._parse_rest_of_l_value_expression(subscript_exp, tokenizer)

    def _parse_new_field_expression(
        self, tokenizer: TigerLexer
    ) -> Union[LValueExpression, AssignmentExpression]:
        """
        fieldExp → lValue . id
        """
        left_id_token = tokenizer.next()
        _assert_identifier(left_id_token)
        _assert_tkn_val(tokenizer.next(), Punctuation.DOT)
        right_id_token = tokenizer.next()
        _assert_identifier(right_id_token)
        field_exp = LValueExpression(
            FieldExpression(
                LValueExpression(Identifier(left_id_token.value)),
                Identifier(right_id_token.value),
            )
        )
        return self._parse_rest_of_l_value_expression(field_exp, tokenizer)

    def _parse_rest_of_l_value_expression(
        self, l_value: LValueExpression, tokenizer: TigerLexer
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
        while tokenizer.peek() in [Punctuation.BRACKET_OPEN, Punctuation.DOT]:
            next_token = tokenizer.next()
            if next_token == Punctuation.BRACKET_OPEN:
                exp = self._parse_expression(tokenizer)
                l_value = LValueExpression(SubscriptExpression(l_value, exp))
                _assert_tkn_val(tokenizer.next(), Punctuation.BRACKET_CLOSE)
            elif next_token == Punctuation.DOT:
                id_token = tokenizer.next()
                l_value = LValueExpression(
                    FieldExpression(l_value, Identifier(id_token.value))
                )

        if tokenizer.peek() == Punctuation.ASSIGNMENT:
            return self._parse_assignment_expression(l_value, tokenizer)
        else:
            return l_value

    def _parse_assignment_expression(
        self,
        l_value: LValueExpression,
        tokenizer: TigerLexer,
    ) -> AssignmentExpression:
        """
        assignment → lValue := exp
        """
        _assert_tkn_val(tokenizer.next(), Punctuation.ASSIGNMENT)
        exp = self._parse_expression(tokenizer)
        return AssignmentExpression(l_value, exp)

    def _parse_call_expression(
        self,
        tokenizer: TigerLexer,
    ) -> CallExpression:
        """
        callExp → id ( exp∗, )
        """
        id_token = tokenizer.next()
        _assert_identifier(id_token)
        _assert_tkn_val(tokenizer.next(), Punctuation.PAREN_OPEN)
        exps = []
        next_token = tokenizer.next()
        while True:
            if next_token == Punctuation.PAREN_CLOSE:
                break
            exps.append(self._parse_expression(tokenizer))
            next_token = tokenizer.next()

        return CallExpression(Identifier(id_token.value), exps)

    def _parse_rec_create_expression(
        self,
        tokenizer: TigerLexer,
    ) -> RecCreateExpression:
        """
        recCreate → tyId { fieldCreate∗, }
        """
        id_token = tokenizer.next()
        _assert_identifier(id_token)
        _assert_tkn_val(tokenizer.next(), Punctuation.CURLY_OPEN)
        field_creates = []
        next_token = tokenizer.next()
        while True:
            if next_token == Punctuation.CURLY_CLOSE:
                break
            field_creates.append(self._parse_field_create(tokenizer))
            next_token = tokenizer.next()
        return RecCreateExpression(
            Identifier(id_token.value),
            field_creates,
        )

    def _parse_field_create(
        self,
        tokenizer: TigerLexer
    ) -> FieldCreate:
        id_token = tokenizer.next()
        _assert_tkn_val(tokenizer.next(), OpToken.EQ)
        exp = self._parse_expression(tokenizer)
        return FieldCreate(Identifier(id_token.value), exp)

    def _parse_type_declaration(
        self, tokenizer: TigerLexer
    ) -> TypeDeclaration:
        """
        tyDec → type tyId = ty
        """
        _assert_tkn_val(tokenizer.next(), Keyword.TYPE)

        id_token = tokenizer.next()
        _assert_identifier(id_token)

        _assert_tkn_val(tokenizer.next(), OpToken.EQ)

        next_token = tokenizer.peek()
        if next_token.is_identifier:
            type_node = TypeIdentifier(next_token.value)
        elif next_token == Keyword.ARRAY:
            type_node = self._parse_array_type(tokenizer)
        elif next_token == Punctuation.CURLY_OPEN:
            type_node = self._parse_record_type(tokenizer)
        else:
            raise ParserException(
                'Expected type identifier, array type, or record type. '
                'Found "{}"'.format(next_token)
            )
        return TypeDeclaration(TypeIdentifier(id_token.value), type_node)

    def _parse_array_type(self, tokenizer: TigerLexer) -> ArrayType:
        """
        arrTy → array of tyId
        """
        _assert_tkn_val(tokenizer.next(), Keyword.ARRAY)
        _assert_tkn_val(tokenizer.next(), Keyword.OF)
        type_id_token = tokenizer.next()
        return ArrayType(TypeIdentifier(type_id_token.value))

    def _parse_record_type(self, tokenizer: TigerLexer) -> RecType:
        """
        recTy → { fieldDec∗, }
        """
        _assert_tkn_val(tokenizer.next(), Punctuation.CURLY_OPEN)
        field_decs = []
        next_token = tokenizer.next()
        while True:
            if next_token == Punctuation.CURLY_OPEN:
                break
            field_decs.append(self._parse_field_declaration(tokenizer))
            next_token = tokenizer.next()
        return RecType(field_decs)

    def _parse_func_declaration(
        self, tokenizer: TigerLexer
    ) -> FunDeclaration:
        """
        funDec → function id ( fieldDec∗, ) = exp
                | function id ( fieldDec∗, ) : tyId = exp
        """
        _assert_tkn_val(tokenizer.next(), Keyword.FUNCTION)

        id_token = tokenizer.next()
        _assert_identifier(id_token)

        _assert_tkn_val(tokenizer.next(), Punctuation.PAREN_OPEN)
        next_token = tokenizer.next()
        field_decs = []
        while next_token != Punctuation.PAREN_CLOSE:
            field_decs.append(self._parse_field_declaration(tokenizer))
            next_token = tokenizer.next()  # either comma or close paren

        type_id_node = self._get_type_annotation(tokenizer)

        _assert_tkn_val(tokenizer.next(), OpToken.EQ)
        exp_node = self._parse_expression(tokenizer)
        return FunDeclaration(
            Identifier(id_token.value),
            field_decs,
            exp_node,
            type_identifier=type_id_node,
        )

    def _parse_var_declaration(
        self,
        tokenizer: TigerLexer,
    ) -> VarDeclaration:
        """
        varDec → var id := exn
                | var id : tyId := exp
        """
        _assert_tkn_val(tokenizer.next(), Keyword.VAR)

        id_token = tokenizer.next()
        _assert_identifier(id_token)

        type_id = self._get_type_annotation(tokenizer)

        _assert_tkn_val(tokenizer.next(), Punctuation.ASSIGNMENT)
        exp_node = self._parse_expression(tokenizer)

        return VarDeclaration(
            Identifier(id_token.value),
            exp_node,
            type_identifier=type_id,
        )

    def _parse_field_declaration(
        self, tokenizer: TigerLexer
    ) -> FieldDeclaration:
        """
        fieldDec → id : tyId
        """
        id_token = tokenizer.next()
        _assert_identifier(id_token)

        type_id = self._get_type_annotation(tokenizer)
        if type_id is None:
            raise ParserException(
                'Field declarations require type annotation.'
            )
        return FieldDeclaration(Identifier(id_token.value), type_id)
