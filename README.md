# ATL Automaton Generator

## Overview
This repository contains a Python implementation for parsing, transforming, and verifying **Alternating-time Temporal Logic (ATL)** formulas, along with the generation of **Automata over Concurrent Game Structures (ACG)** based on valid ATL formulas.

## Features
- **Tokenization & Parsing:** Converts ATL formulas into an Abstract Syntax Tree (AST).
- **Transformation to Fundamental Operators:** Rewrites formulas using a fundamental set of temporal operators.
- **Formula Filtering:** Determines if a given formula is strictly ATL, ATL*, or invalid.
- **ACG Construction:** Builds an automaton from valid ATL formulas, including:
  - Extraction of atomic propositions.
  - Computation of the closure set.
  - Definition of transition rules between states.
- **Automated Testing:** The main script executes a set of test formulas and prints the corresponding results.

## Installation
Ensure you have Python 3 installed and clone the repository:
```bash
$ git clone https://github.com/your-repo/ATL-Automaton.git
$ cd ATL-Automaton
```

## Usage
Run the main script to analyze ATL formulas and generate their corresponding automata when applicable:
```bash
$ python main.py
```

## Expected Output
For each test formula, the script outputs:
- The original formula.
- Tokenized representation.
- Initial AST structure.
- Transformed AST (using fundamental operators).
- Reconstructed formula from the transformed AST.
- Filter classification (**ATL**, **ATL* but not ATL**, or **Invalid**).
- If the formula is valid in ATL, the generated **ACG representation** is displayed.

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

## Future Work
- Implement support for additional ATL operators.
- Improve filtering conditions for edge cases.
- Extend automaton capabilities for model checking.

## License
This project is licensed under the MIT License.

---

For contributions or issues, feel free to open a pull request or report an issue on GitHub.

