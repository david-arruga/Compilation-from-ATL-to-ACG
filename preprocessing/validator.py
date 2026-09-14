
from __future__ import annotations

from .ast_nodes import (
    ParseNode, T, F, Var, And, Or, Not, Next, Until, Release,
    Globally, Eventually, Implies, Iff, Modality, DualModality,
)


def validate_core_atl(ast: ParseNode, agents=None) -> None:
    """Check the normalized state grammar implemented by the ACG builder.

    Path negation is not strategic negation. In particular <A> !(p U q)
    and residual dual modalities are rejected, not silently reinterpreted.
    This validates the compiler's fragment, not every possible ATL extension.
    """
    universe = None if agents is None else set(agents)

    def visit(node):
        if isinstance(node, (T, F, Var)):
            return
        if isinstance(node, (And, Or)):
            visit(node.lhs)
            visit(node.rhs)
            return
        if isinstance(node, Not):
            if not isinstance(node.sub, (Var, Modality)):
                raise ValueError("Expected normalized negation before a literal or strategic modality.")
            visit(node.sub)
            return
        if isinstance(node, Modality):
            if not isinstance(node.agents, (list, tuple, set, frozenset)) or not all(
                isinstance(a, str) for a in node.agents
            ):
                raise ValueError("A coalition must be a collection of agent names.")
            if universe is not None and not set(node.agents) <= universe:
                raise ValueError("Coalition contains agents outside the CGS agent set.")
            path = node.sub
            if isinstance(path, (Next, Globally)):
                visit(path.sub)
                return
            if isinstance(path, Until):
                visit(path.lhs)
                visit(path.rhs)
                return
        raise ValueError(
            "Unsupported normalized ATL form: " + type(node).__name__
            + ". Expected state Booleans and <A>X, <A>G or <A>U; "
            "path negation, residual dual modalities and bare temporal operators are not supported."
        )

    visit(ast)


def filter(ast: ParseNode, strict_ATL: bool = True) -> str:
    if strict_ATL:
        try:
            validate_core_atl(ast)
        except ValueError:
            return "UNSUPPORTED"
        return "ATL"
    def validate_structure(node: ParseNode) -> str | None:
        if isinstance(node, (Modality, DualModality)):
            if not isinstance(node.agents, list) or not all(isinstance(a, str) for a in node.agents):
                print("ERROR: Modality must have a list of agent names.")
                return "INVALID"
            if not isinstance(node.sub, ParseNode):
                print("ERROR: Modality must have a valid subformula.")
                return "INVALID"

        if isinstance(node, Until):
            if not isinstance(node.lhs, ParseNode) or not isinstance(node.rhs, ParseNode):
                print("ERROR: Until must have both lhs and rhs as valid subformulas.")
                return "INVALID"

        for value in getattr(node, "__dict__", {}).values():
            if isinstance(value, ParseNode):
                result = validate_structure(value)
                if result:
                    return result
        return None

    def validate_atl_semantics(node: ParseNode, parent: ParseNode | None = None, grandparent: ParseNode | None = None) -> str | None:
        if isinstance(node, (Modality, DualModality)):
            sub = node.sub
            if isinstance(sub, Not):
                sub = sub.sub  
            if not isinstance(sub, (Next, Globally, Until)):
                print("ERROR: Modality must be applied to Next, Globally, or Until (or their negation).")
                return "ATL* but not ATL"

        if isinstance(node, Until):
            if getattr(node, "generated_from_eventually", False):
                pass
            elif isinstance(parent, (Modality, DualModality)):
                pass
            elif isinstance(parent, Not) and isinstance(grandparent, (Modality, DualModality)):
                pass
            else:
                print("ERROR: Until must be directly under a modality (or its negation).")
                return "ATL* but not ATL"

        if isinstance(node, (Next, Globally)):
            if not isinstance(parent, (Modality, DualModality)):
                print(f"ERROR: {node.__class__.__name__} must be directly under a modality.")
                return "ATL* but not ATL"

        for value in getattr(node, "__dict__", {}).values():
            if isinstance(value, ParseNode):
                result = validate_atl_semantics(value, node, parent)
                if result:
                    return result
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ParseNode):
                        result = validate_atl_semantics(item, node, parent)
                        if result:
                            return result
        return None

    structure_result = validate_structure(ast)
    if structure_result:
        return structure_result

    if not strict_ATL:
        return "ATL*"

    semantic_result = validate_atl_semantics(ast)
    if semantic_result:
        return semantic_result

    return "ATL"


__all__ = ["filter"]
