
from __future__ import annotations

from .ast_nodes import (
    ParseNode, T, F, Var, And, Or, Not, Next, Until, Release,
    Globally, Eventually, Implies, Iff, Modality, DualModality,
)


def filter(ast: ParseNode, strict_ATL: bool = True) -> str:
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
