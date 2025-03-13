def transform_to_fundamental(node):

    if isinstance(node, Eventually):
        return Until(T(), transform_to_fundamental(node.sub), generated_from_eventually=True)  

    elif isinstance(node, Release):
        lhs_transformed = transform_to_fundamental(node.lhs)
        rhs_transformed = transform_to_fundamental(node.rhs)
        
        if isinstance(lhs_transformed, Modality):
            return Modality(lhs_transformed.agents, Not(Until(Not(lhs_transformed.sub), Not(rhs_transformed))))
        
        return Not(Until(Not(lhs_transformed), Not(rhs_transformed)))

    elif isinstance(node, And):
        return And(transform_to_fundamental(node.lhs), transform_to_fundamental(node.rhs))

    elif isinstance(node, Or):
        return Or(transform_to_fundamental(node.lhs), transform_to_fundamental(node.rhs))

    elif isinstance(node, Not):
        return Not(transform_to_fundamental(node.sub))

    elif isinstance(node, Next):
        return Next(transform_to_fundamental(node.sub))

    elif isinstance(node, Until):
        return Until(transform_to_fundamental(node.lhs), transform_to_fundamental(node.rhs))


    elif isinstance(node, Globally):
        return Globally(transform_to_fundamental(node.sub))

    elif isinstance(node, Modality):
        return Modality(node.agents, transform_to_fundamental(node.sub))

    return node
