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
Original Formula : <A> next (p and q)
Tokens: [(5, '<'), (19, 'A'), (6, '>'), (15, 'next'), (1, '('), (18, 'p'), (8, 'and'), (18, 'q'), (2, ')')]
Initial AST:
Modality (A)
    Next
        And
            Var('p')
            Var('q')
Transformed AST:
Modality (A)
    Until
        T
        And
            Var('p')
            Var('q')
Reconstructed Formula : <A> (⊤ U (p AND q))
Filter result : ATL

ACG(
  Alphabet: [{}, {p}, {q}, {p, q}],
  States: ['<A> (⊤ U (p AND q))', 'p AND q', 'p', 'q'],
  Initial State: <A> (⊤ U (p AND q)),
  Transitions:
    δ(p, {p}) → ⊤
    δ(p, {q}) → ⊥
    δ(q, {p}) → ⊥
    δ(q, {q}) → ⊤
    δ(p AND q, {p, q}) → ( ( p, ε ) AND ( q, ε ) )
    δ(<A> (⊤ U (p AND q)), {p, q}) → ( <A> (⊤ U (p AND q)), ◇, {A} )
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

