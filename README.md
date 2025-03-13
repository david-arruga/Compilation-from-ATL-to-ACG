# ATL Automaton Generator

## Currently implemented
- **Tokenization**
- **Parsing into AST**
- **Transformation to Fundamental AST:** 
- **Filtering:** 
- **ACG Construction:** 
  - Extraction of atomic propositions.
  - Computation of the closure set.
  - Definition of transition function.


## Expected Output
For each test formula, the script outputs:
- The original formula.
- Tokenized representation.
- Initial AST structure.
- Transformed AST.
- Reconstructed formula from the transformed AST.
- Filter classification (**ATL**, **ATL* **, or **Invalid**).
- If the formula is ATL, the generated **ACG representation** is displayed.

### Example Output
```
======================================================================
 Original Formula : <A> eventually (<A> p until q)
 Tokens: [(5, '<'), (19, 'A'), (6, '>'), (17, 'eventually'), (1, '('), (5, '<'), (19, 'A'), (6, '>'), (18, 'p'), (14, 'until'), (18, 'q'), (2, ')')]
 Initial AST :
Modality (A)
    Eventually
        Modality (A)
            Until
                Var('p')
                Var('q')
 Transformed AST :
Modality (A)
    Until
        T
        Modality (A)
            Until
                Var('p')
                Var('q')
 Reconstructed Formula : <A> (⊤ U <A> (p U q))
 Filter result :  ATL
ACG(
  Alphabet: ['{p, q}', '{p}', '{q}', '{}'],
  States: ['<A> (p U q)', '<A> (⊤ U <A> (p U q))', 'p', 'q', '⊤'],
  Initial State: <A> (⊤ U <A> (p U q)),
  Transitions:
    δ(<A> (p U q), {}) → (( q, ε ) OR (( p, ε ) AND ( <A> (p U q), ◇, {A} )))
    δ(<A> (p U q), {p, q}) → (( q, ε ) OR (( p, ε ) AND ( <A> (p U q), ◇, {A} )))
    δ(<A> (p U q), {q}) → (( q, ε ) OR (( p, ε ) AND ( <A> (p U q), ◇, {A} )))
    δ(<A> (p U q), {p}) → (( q, ε ) OR (( p, ε ) AND ( <A> (p U q), ◇, {A} )))
    δ(<A> (⊤ U <A> (p U q)), {}) → (( <A> (p U q), ε ) OR (( ⊤, ε ) AND ( <A> (⊤ U <A> (p U q)), ◇, {A} )))
    δ(<A> (⊤ U <A> (p U q)), {p, q}) → (( <A> (p U q), ε ) OR (( ⊤, ε ) AND ( <A> (⊤ U <A> (p U q)), ◇, {A} )))
    δ(<A> (⊤ U <A> (p U q)), {q}) → (( <A> (p U q), ε ) OR (( ⊤, ε ) AND ( <A> (⊤ U <A> (p U q)), ◇, {A} )))
    δ(<A> (⊤ U <A> (p U q)), {p}) → (( <A> (p U q), ε ) OR (( ⊤, ε ) AND ( <A> (⊤ U <A> (p U q)), ◇, {A} )))
    δ(p, {}) → ⊥
    δ(p, {p, q}) → ⊤
    δ(p, {q}) → ⊥
    δ(p, {p}) → ⊤
    δ(q, {}) → ⊥
    δ(q, {p, q}) → ⊤
    δ(q, {q}) → ⊤
    δ(q, {p}) → ⊥
)
======================================================================
```


