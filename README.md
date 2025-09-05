
# Compilation from ATL to ACG

A modular reference implementation for the end‑to‑end compilation and verification pipeline:

**Parsing → Normalisation → ACG construction → Acceptance Game → Büchi solving**,
including four example Concurrent Game Structures (CGS) and two benchmarking suites
(random ATL formulae; parametric families based on light‑switch controllers).

> This repository accompanies a computational logic thesis. The code is organized to make
> each mathematical stage explicit, testable, and reusable.

---

## 1. Scientific Overview

Given an ATL/ATL\* formula φ and a Concurrent Game Structure G, this project builds a
*compact ACG* (automaton over CGS‑game formulas) whose transitions encode symbolic
obligations (ε, □, ◇ atoms), forms the product Acceptance Game A = ACG × G,
then solves a Büchi objective over A to determine whether the initial product state belongs to
Player 0’s winning region. If so, φ is satisfiable on G.

Mathematical components reflected in the code:

1. **Syntactic processing** (tokenization, parsing to AST).
2. **Normalisation** (modal dualities, elimination of Release and Eventually → Until, NNF).
3. **ACG construction** (closure of subformulas; compact δ with wildcard; lazy evaluation for
   atomic states p and ¬p against labels σ ⊆ AP).
4. **Acceptance Game** (layered arena with state → atom\_selection → atom\_applied → …,
   partition into S1/S2 and Büchi set B).
5. **Büchi solver** (classical predecessor/attractor iteration yielding the winning set Sj).

---

## 2. Repository Layout

```
.
├── main.py                                  # End‑to‑end driver (edit formula/CGS as desired)
├── preprocessing/                           # Lexing, AST, parsing, normalisation, ATL filter
│   ├── __init__.py
│   ├── tokens.py
│   ├── ast_nodes.py
│   ├── parser.py
│   ├── transformer.py
│   └── validator.py
├── acg/                                     # ACG core and builders
│   ├── __init__.py
│   ├── cgs.py                               # CGS class used across the pipeline
│   ├── acg_core.py                          # Universal/Existential/Epsilon atoms, ACG
│   ├── build.py                             # Closure, transitions, size utilities
│   └── utils.py                             # Helpers for atomic δ and analysis
├── acceptance_game/                         # Product ACG×CGS and expansion routines
│   ├── __init__.py
│   ├── product.py                           # GameProduct container and pretty printing
│   ├── expanders.py                         # State expansion (state/atom_selection/…)
│   ├── utils.py
│   └── examples.py                          # cgs1..cgs4 used in the thesis
├── büchi_solver/                            # Predecessors, attractors, Büchi solver
│   ├── __init__.py
│   ├── predecessors.py
│   ├── attractor.py
│   └── solver.py
├── benchmarks/
│   ├── random_generator/
│   │   ├── __init__.py
│   │   └── generator.py                     # Valid ATL formulae by exact depth
│   └── parametric_families/
│       ├── __init__.py
│       ├── cgs_factory.py                   # Light‑switch CGS with n controllers
│       ├── families_x.py                    # Families using Next
│       ├── families_g.py                    # Families using Globally
│       └── families_u.py                    # Families using Until
├── smoke_preprocessing.py                   # Sanity test of preprocessing
├── smoke_acg.py                             # ACG construction sanity test
├── smoke_acceptance_game.py                 # Product construction sanity test
├── smoke_solver.py                          # End‑to‑end (ACG×CGS + Büchi)
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

> Note: The package directory name `büchi_solver/` uses a non‑ASCII character. On systems
> where this is problematic, use a consistent ASCII spelling (e.g. `buchi_solver`) **throughout**
> the folder name and imports.

---

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

