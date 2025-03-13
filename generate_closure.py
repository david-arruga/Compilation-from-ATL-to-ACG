def generate_closure(ast):
    closure = set()

    def is_subformula(node):
        
        if node == ast:
            return True

        if isinstance(node, Var):
            return True

        if isinstance(node, Modality):
            return True

        if isinstance(node, (Next, Globally)):
            return False

        if isinstance(node, Until) and not isinstance(node, Modality):
            return False

        return True  


    def traverse(node):
        
        if is_subformula(node):
            closure.add(node)

        for child in getattr(node, "__dict__", {}).values():
            if isinstance(child, ParseNode):
                traverse(child)

    traverse(ast)
    return closure
