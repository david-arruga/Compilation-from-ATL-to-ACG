
# Compilation from ATL to ACG

This repository contains the research software implementing the technical results of the associated thesis ([PDF](./docs/thesis.pdf)). The tool checks whether coalitions of agents can guarantee temporal goals in interactive systems. Given a Concurrent Game Structure (CGS) and an Alternating-time Temporal Logic (ATL) formula, it runs an end-to-end compilation pipeline: parsing and normalisation → construction of an Automaton over Concurrent Game Structures (ACG) → product acceptance game → Büchi game solving to decide acceptance of the formula on the model. The work provides an alternative to traditional ATL model checking by introducing a compilation approach based on ACGs. The repository also includes four reference CGSs and two benchmarking suites (random ATL formulae and parametric light-switch families).

## Algorithmic Overview

Given an ATL formula φ and a Concurrent Game Structure (CGS) C, the tool constructs an Automaton over Concurrent Game Structures (ACG), forms the product acceptance game ACG × C, and solves a Büchi game to decide acceptance of φ on C. Each stage mirrors the underlying mathematics:

1. **Preprocessing (preprocessing/)**  
   The input ATL formula is tokenized and parsed into an AST. We then apply modal dualities (to eliminate dual boxes via negation) and rewrite derived operators so that `Release` and `Eventually` are expressed using the core temporal connectives (`Until`, `Next`). Finally, we push negations to obtain Negation Normal Form. The result is a normalized AST φ′ semantically equivalent to the original φ.

2. **ACG Construction (acg/)**  
   From φ′ we extract the set of atomic propositions AP and compute the closure Cl(φ′), consisting of subformulas and their negations. The ACG uses states Q = Cl(φ′), initial state q₀ = φ′, and (optionally) alphabet Σ ⊆ 2^AP. Transitions δ are specified compositionally: Boolean nodes map to conjunction/disjunction over ε-atoms; temporal/modality nodes map to **UniversalAtom**/**ExistentialAtom** obligations over sub-states and coalitions. Atomic states `p` and `¬p` are evaluated lazily against the current label σ ⊆ AP. The Büchi acceptance set F ⊆ Q corresponds to the standard acceptance conditions induced by the normalized temporal patterns present in φ′.

3. **Acceptance Game (acceptance_game/)**  
   We build the product arena whose positions track both the automaton state and the CGS state, together with intermediate “choice” layers that encode the disjunctive/conjunctive structure of δ and the alternation of strategic quantifiers. The arena is partitioned into S₁ (Prover) and S₂ (Refuter); the Büchi set B lifts the automaton acceptance F to product positions. Edges are induced by δ-expansion and by CGS transitions under joint actions of agents.

4. **Büchi Solving (büchi_solver/)**  
   A classical predecessor/attractor iteration computes the winning set Sⱼ for S₁ in the Büchi game. Acceptance of φ on C holds exactly when the initial product position belongs to Sⱼ.

This pipeline establishes a compilation-based alternative to traditional ATL model checking, replacing on-the-fly evaluation with a structured construction: **formula → ACG → acceptance game → Büchi solution**.

## Repository Layout

```text
.
├── README.md
├── main.py
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── .gitattributes
├── docs/
│   └── thesis.pdf
├── preprocessing/
│   ├── __init__.py
│   ├── tokens.py
│   ├── ast_nodes.py
│   ├── parser.py
│   ├── transformer.py
│   └── validator.py
├── acg/
│   ├── __init__.py
│   ├── cgs.py
│   ├── acg_core.py
│   ├── build.py
│   └── utils.py
├── acceptance_game/
│   ├── __init__.py
│   ├── product.py
│   ├── expanders.py
│   ├── utils.py
│   └── examples.py
├── buchi_solver/
│   ├── __init__.py
│   ├── predecessors.py
│   ├── attractor.py
│   └── solver.py
├── benchmarks/
│   ├── random_generator/
│   │   ├── __init__.py
│   │   └── generator.py
│   └── parametric_families/
│       ├── __init__.py
│       ├── cgs_factory.py
│       ├── families_x.py
│       ├── families_g.py
│       └── families_u.py
└── smokes/
    ├── smoke_preprocessing.py
    ├── smoke_acg.py
    ├── smoke_acceptance_game.py
    └── smoke_solver.py
```


## 3. Software Requirements

* Python ≥ 3.10
* NumPy, Pandas, Matplotlib, NetworkX, SciPy

Install into a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` (minimal):

```
numpy
pandas
matplotlib
networkx
scipy
```

---

## 4. Quick Start

### 4.1 End‑to‑End (default example)

Run the full pipeline using the example formula and `cgs1` defined in `acceptance_game/examples.py`:

```bash
python main.py
```

The script prints the AST, a tree rendering, ACG size, the compact transition relation, the
acceptance product summary, and the Büchi winner. If the initial product state lies in Player 0’s
winning set `Sj`, the formula is satisfiable on the chosen CGS.

> If you prefer to hard‑code a different formula/CGS, edit `main.py` accordingly (or add CLI
> flags to pass `--formula` and `--cgs`).

### 4.2 Sanity Checks (smoke tests)

* Preprocessing only:

  ```bash
  python smoke_preprocessing.py
  ```
* ACG construction:

  ```bash
  python smoke_acg.py
  ```
* Product ACG×CGS:

  ```bash
  python smoke_acceptance_game.py
  ```
* Büchi solving end‑to‑end:

  ```bash
  python smoke_solver.py
  ```

---

## 5. Pipeline Details

### 5.1 Preprocessing (`preprocessing/`)

* **tokens.py** defines token IDs and `SYMBOL_MAP` for temporal symbols.
* **ast\_nodes.py** defines the node classes (`Var`, `And`, `Or`, `Not`, `Next`, `Globally`,
  `Until`, `Modality`, `DualModality`, etc.).
* **parser.py** implements `tokenize` → `parse` to obtain an AST.
* **transformer.py** implements `apply_modal_dualities`, `eliminate_f_and_r`,
  `push_negations_to_nnf`, and `normalize_formula`.
* **validator.py** implements the ATL/ATL\* filter used by the benchmarks.

### 5.2 ACG (`acg/`)

* **acg\_core.py** defines the automaton atoms `EpsilonAtom`, `UniversalAtom`, `ExistentialAtom`,
  and the `ACG` container with compact δ (wildcard `None`).
* **build.py** computes the closure, marks initial/final states, builds transitions, and patches
  `get_transition` to lazily resolve atomic states against σ.
* **cgs.py** provides the `CGS` model with propositions, agents, decisions, labeling and a
  deterministic transition function.

### 5.3 Acceptance Game (`acceptance_game/`)

* **product.py** defines `GameProduct` and text rendering of product nodes.
* **expanders.py** expands nodes from `state` to `atom_selection`, `atom_applied`, etc.
* **examples.py** provides four CGS instances (`cgs1..cgs4`) used in the thesis.

### 5.4 Büchi Solver (`büchi_solver/`)

* **predecessors.py** implements `predecessor_1/2`.
* **attractor.py** implements `attractor_1/2`.
* **solver.py** implements `solve_buchi_game`, returning the winning region `Sj` and the
  cumulative avoid set `W_total`.

---

## 6. Benchmarks

### 6.1 Random ATL Formulae (`benchmarks/random_generator/`)

Generates structurally valid ATL formulas with exact depth over a chosen CGS. The generator
samples from `Var`, Boolean connectives, and modality‑temporal forms (`⟨A⟩ X`, `⟨A⟩ G`,
`⟨A⟩ (· U ·)`), normalises them, and accepts those classified as ATL by the filter.

Usage pattern:

```python
from benchmarks.random_generator.generator import (
    generate_random_valid_atl_formula, generate_valid_formulas_by_depth,
)
from acceptance_game.examples import cgs1

f = generate_random_valid_atl_formula(cgs1, depth=5)
print(f.to_formula())
```

For reproducible samples, set `random.seed(...)` at the top of your script.

### 6.2 Parametric Families (`benchmarks/parametric_families/`)

A family of CGS instances with `n` independent controllers (`ctrl_0..ctrl_{n-1}`) toggling
Boolean switches with propositions `p_i`. The modules `families_x.py`, `families_g.py`, and
`families_u.py` construct flat, disjunctive, individual, nested, and (stepwise) negated families
for the `Next`, `Globally`, and `Until` operators respectively.

Usage pattern:

```python
from benchmarks.parametric_families.cgs_factory import generate_lights_cgs
from benchmarks.parametric_families.families_g import generate_flatG_spec
from preprocessing import normalize_formula
from acg import build_acg_final

G = generate_lights_cgs(n=6)
phi = generate_flatG_spec(n=6)
phi = normalize_formula(phi)
ACG = build_acg_final(phi, G, materialize_alphabet=False)
```

---

## 7. Reproducibility & Performance Notes

* **Alphabet materialisation**: set `materialize_alphabet=True` in ACG construction to list
  `2^AP` explicitly (useful for inspection); otherwise δ uses a wildcard and resolves atomic
  states lazily against the CGS labeling.
* **Complexity**: product and Büchi solving are sensitive to the size of the closure and branching
  of joint actions in the CGS; prefer minimal AP sets and avoid redundant modalities in benchmarks.
* **Seeding**: the random generator does not seed by default; set a fixed seed for fairness.

---

## 8. How to Cite

Please cite this repository as:

```
@software{Compilation_from_ATL_to_ACG,
  author  = {Arruga, David},
  title   = {Compilation from ATL to ACG: A Modular Pipeline},
  year    = {2025},
  url     = {https://github.com/david-arruga/Compilation-from-ATL-to-ACG}
}
```

A `CITATION.cff` file is included for GitHub’s citation widget.

---

## 9. License

This project is released under the terms of the LICENSE file included in this repository.

---

## 10. Contact

For questions or clarifications regarding the code or the accompanying thesis, please open an
issue on GitHub or contact the author through the email listed in the thesis manuscript.

