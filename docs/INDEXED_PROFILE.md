# Indexed constructor: profiling diagnosis

Run: `PYTHONHASHSEED=0 python3 -m scripts.profile_indexed`.

This supplements INDEXED_SCALING.md; production compiler unchanged. Forty timed runs with GC callbacks (five repeats per size/mode, randomized and interleaved), plus four separate cProfile runs. All inputs are already-normalized nested G formulas with one agent and one proposition. Timing excludes input generation, explicit pre-run GC and output destruction. Callback/profiler instrumentation has overhead; these are diagnostic measurements, not replacements for the uninstrumented benchmark.

| AST nodes | Median total, GC on (ms) | Median GC pauses (ms) | Median total, GC off (ms) | cProfile calls |
|---:|---:|---:|---:|---:|
| 8193 | 30.20 | 7.07 | 23.86 | 139281 |
| 16385 | 104.56 | 53.28 | 60.79 | 278545 |
| 32769 | 318.82 | 198.99 | 101.62 | 557073 |
| 65537 | 706.37 | 465.25 | 327.70 | 1114129 |

## Findings

- GC pauses directly account for a substantial part of elapsed time at larger sizes. Medians in separate columns need not add or subtract to the median of paired differences.
- For these four profiled inputs, recorded call counts are exactly 17N. This is an empirical count for this family and profiler, not a count of all machine operations or a universal bound.
- No recursive formula hashing, copying or rendering appears in these profiles. Source inspection confirms the compiler uses integer references, bounded-arity transition expressions and iterative traversal.
- A 65,537-node input completed. This extends the measured range; it does not establish the maximum feasible size.
- Non-GC time per node still varies. These experiments do not isolate allocation costs, cache effects, container resizing or shared-host scheduling; attributing the residual to one of them would be speculation.

## Cost-model qualifications

Each syntax occurrence is visited once; child extraction touches at most two references, and transition expressions have bounded size. Coalition validation/construction and complement require work depending on the coalition representation and agent universe. The O(N(1+a)) description assumes coalition encodings have length O(a), identifiers have unit cost, and hash containers have expected/amortized constant operation cost. A list with arbitrarily many duplicate agent names violates the first assumption: actual serialized coalition length must then be charged. String hashing/length and arbitrary-precision IDs also matter under a bit-cost model. Explicit alphabet enumeration and diagnostic label reconstruction remain outside compilation.

## Decision

There is no identified repeated-subformula traversal to repair in the indexed compiler. Keep production behavior unchanged: globally disabling automatic GC inside a library constructor would affect the caller and is not justified by this benchmark. The control only disables it temporarily in the experiment and restores its previous state. The evidence supports a linear structural-work account with the stated assumptions, while actual Python wall-clock proportionality remains approximate and environment dependent. This does not alter semantic correctness claims or constitute a Lean certification of Python runtime.
