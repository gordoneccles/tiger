from abc import ABC
from contextlib import contextmanager
from typing import List

from tiger_pl.parser import (
    Program, Expression, TypeIdentifier, Identifier, TigerType as TigerTypeNode,
    Declaration, IfThenExpression, IfThenElseExpression, ForExpression,
    WhileExpression, SeqExpression, NegationExpression, CallExpression,
)


class TigerType(ABC):
    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not(self == other)


class IntegerType(TigerType):
    pass


class StringType(TigerType):
    pass


class VoidType(TigerType):
    pass


class DeclaredType(TigerType):
    def __init__(self, tiger_type_node: TigerTypeNode):
        self._tiger_type_node = tiger_type_node

    def __eq__(self, other):
        return (
            isinstance(other, DeclaredType)
            and self._tiger_type_node == other._tiger_type_node
        )


class TigerTypeError(Exception):

    @classmethod
    def for_condition(cls, actual: TigerType) -> "TigerTypeError":
        return cls(f'Condition clauses must return an integer type. Got {actual}.')

    @classmethod
    def for_void(cls, exp_name: str, actual: TigerType) -> "TigerTypeError":
        return cls(f'{exp_name} may not return anything. Got {actual}.')

    @classmethod
    def for_annotation(
        cls, expected: TigerType, actual: TigerType
    ) -> "TigerTypeError":
        return cls(f'Expected {expected}, got {actual}')


class UnknownIdentifierException(Exception):
    def __init__(self, id_type: str, identifier: Identifier):
        super().__init__(f'Unknown {id_type} "{identifier}"')


STRING = 'string'
INT = 'int'
VOID = 'void'


class Scope(object):
    def __init__(self, outer_scope: "Scope" = None):
        self.outer_scope = outer_scope
        # We _could_ just include the base types in the outermost scope, but there's no
        # harm in including them in every subscope as well. Plus it saves us from having
        # to walk the linked list every time they appear, which is probably a lot.
        self._types = {
            STRING: StringType(),
            INT: IntegerType(),
            VOID: VoidType(),
        }
        self._vars = {}
        self._funs = {}

    def add_type(self, type_identifier: TypeIdentifier, tiger_type: DeclaredType):
        self._types[type_identifier] = tiger_type

    def add_var(self, identifier: Identifier, tiger_type: TigerType):
        self._vars[identifier] = tiger_type

    def add_func(
        self,
        identifier: Identifier,
        return_type: TigerType,
        param_types: List[TigerType],
    ):
        self._funs[identifier] = (return_type, param_types)

    def get_var_type(self, identifier: Identifier) -> TigerType:
        if identifier in self._vars:
            return self._vars[identifier]
        elif self.outer_scope is not None:
            return self.outer_scope.get_var_type(identifier)
        raise UnknownIdentifierException('variable', identifier)

    def get_fun_type(self, identifier: Identifier) -> TigerType:
        if identifier in self._funs:
            return self._funs[identifier][0]
        elif self.outer_scope is not None:
            return self.outer_scope.get_fun_type(identifier)
        raise UnknownIdentifierException('function', identifier)
        
    def get_fun_param_types(self, identifier: Identifier) -> List[TigerType]:
        if identifier in self._funs:
            return self._funs[identifier][1]
        elif self.outer_scope is not None:
            return self.outer_scope.get_fun_param_types(identifier)
        raise UnknownIdentifierException('function', identifier)
    
    def is_function(self, identifier: Identifier) -> bool:
        if identifier in self._funs:
            return True
        elif self.outer_scope is not None:
            return self.outer_scope.is_function(identifier)
        return False


class TigerTypeChecker(object):

    def __init__(self, ast: Program):
        self._ast = ast
        self._scope = Scope()

    @contextmanager
    def _new_scope(self):
        self._scope = Scope(self._scope)
        yield
        self._scope = self._scope.outer_scope

    def check(self):
        self._check_expression(self._ast.expression)

    def _check_expression(self, exp: Expression) -> TigerType:
        if exp.is_integer_literal():
            return IntegerType()
        elif exp.is_string_literal:
            return StringType()
        elif exp.is_let:
            return self._check_let_expression(exp)
        elif exp.is_if_then:
            return self._check_if_then_expression(exp)
        elif exp.is_if_then_else:
            return self._check_if_then_else_expression(exp)
        elif exp.is_for:
            return self._check_for_expression(exp)
        elif exp.is_while:
            return self._check_while_expression(exp)
        elif exp.is_seq:
            return self._check_seq_expression(exp)
        elif exp.is_negation:
            return self._check_negation_expression(exp)
        elif exp.is_call:
            return self._check_call_expression(exp)

    def _check_let_expression(self, let_exp: Expression) -> TigerType:
        with self._new_scope():
            for dec in let_exp.declarations:
                self._add_declaration(dec)

            if len(let_exp.expressions) == 0:
                return VoidType()
            else:
                return self._check_expression(let_exp.expressions[-1])

    def _check_if_then_expression(self, if_then_exp: IfThenExpression) -> TigerType:
        cond_type = self._check_expression(if_then_exp.condition)
        if cond_type != IntegerType():
            raise TigerTypeError.for_condition(cond_type)

        cons_type = self._check_expression(if_then_exp.consequent)
        if cons_type != VoidType():
            raise TigerTypeError.for_void('ifThen expressions', cons_type)

        return VoidType()

    def _check_if_then_else_expression(
        self, if_then_else_exp: IfThenElseExpression
    ) -> TigerType:
        cond_type = self._check_expression(if_then_else_exp.condition)
        if cond_type != IntegerType():
            raise TigerTypeError.for_condition(cond_type)

        cons_type = self._check_let_expression(if_then_else_exp.consequent)
        alt_type = self._check_let_expression(if_then_else_exp.alternative)
        if cons_type != alt_type:
            raise TigerTypeError(
                f'The two clauses of an ifThenElse expression must return the same '
                f'type. Got {cons_type} (if clause) and {alt_type} (else clause).'
            )

        return cons_type

    def _check_for_expression(self, for_exp: ForExpression) -> TigerType:
        lower_type = self._check_expression(for_exp.lower_bound)
        upper_type = self._check_expression(for_exp.upper_bound)
        if not(lower_type == upper_type == IntegerType()):
            raise TigerTypeError(
                f'The bounds of a forExp must be integers. '
                f'Got {lower_type} (lower) and {upper_type} (upper).'
            )

        var_type = self._scope.get_var_type(for_exp.identifier)
        if var_type != IntegerType():
            raise TigerTypeError(
                f'The iterating variable of a forExp must be an integer. '
                f'Got {var_type}.'
            )

        body_type = self._check_expression(for_exp.body)
        if body_type != VoidType():
            raise TigerTypeError.for_void('A forExp expression', body_type)

        return VoidType()

    def _check_while_expression(self, while_exp: WhileExpression) -> TigerType:
        cond_type = self._check_expression(while_exp.condition)
        if cond_type != IntegerType():
            raise TigerTypeError.for_condition(cond_type)

        cons_type = self._check_expression(while_exp.consequent)
        if cons_type != VoidType():
            raise TigerTypeError.for_void('ifThen expressions', cons_type)

        return VoidType()

    def _check_seq_expression(self, seq_exp: SeqExpression) -> TigerType:
        if len(seq_exp.expressions) == 0:
            return VoidType()
        else:
            return self._check_expression(seq_exp.expressions[-1])

    def _check_negation_expression(self, neg_exp: NegationExpression) -> TigerType:
        op_type = self._check_expression(neg_exp.expression)
        if op_type != IntegerType():
            raise TigerTypeError(
                f'Negation can only be applied to integer. Got {op_type}.'
            )
        return IntegerType()

    def _check_call_expression(self, call_exp: CallExpression) -> TigerType:
        if not self._scope.is_function(call_exp.identifier):
            raise TigerTypeError(f'"{call_exp.identifier}" is not a function.')

        param_types = self._scope.get_fun_param_types(call_exp.identifier)
        n_provided = len(call_exp.expression)
        n_required = len(param_types)
        if n_provided != n_required:
            raise TigerTypeError(
                f'Function {call_exp.identifier} takes {n_required} arguments, '
                f'{n_provided} given.'
            )

        for pos, exp in enumerate(call_exp.expressions):
            actual_type = self._check_expression(exp)
            formal_type = param_types[pos]
            if actual_type != formal_type:
                raise TigerTypeError(
                    f'Function {call_exp.identifier} expected argument of type '
                    f'{formal_type} at position {pos}, got {actual_type}.'
                )

        return self._scope.get_fun_type(call_exp.identifier)

    def _add_declaration(self, dec: Declaration):
        if dec.is_type:
            self._scope.add_type(dec.type_identifier, DeclaredType(dec.tiger_type))
        elif dec.is_var:
            ret_type = self._check_expression(dec.expression)
            expected = dec.type_identifer
            if expected is not None and ret_type != expected:
                raise TigerTypeError.for_annotation(expected, ret_type)
            self._scope.add_var(dec.identifier, ret_type)
        elif dec.is_fun:
            param_types = [f.type_identifier for f in dec.field_declarations]
            ret_type = self._check_expression(dec.expression)
            expected = dec.type_identifer or VoidType()
            if ret_type != expected:
                raise TigerTypeError.for_annotation(expected, ret_type)
            self._scope.add_func(dec.identifier, ret_type, param_types)
        else:
            raise ValueError(f'Unknown declaration "{dec}"')
