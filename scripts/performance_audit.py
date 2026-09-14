"""Small reproducible diagnostics; not a reproduction of historical figures.

Run: PYTHONHASHSEED=0 python3 -m scripts.performance_audit --output docs/performance_sample.json
"""
import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from preprocessing import ParseNode, Var, Modality, Globally, normalize_formula
from acg import CGS, build_acg_final
import acg.builder as builder
from acceptance_game import build_game
from buchi_solver import solve_buchi_game
from benchmarks.parametric_families import generate_lights_cgs, generate_flatG_spec


def node_count(root):
    stack, total = [root], 0
    while stack:
        node = stack.pop(); total += 1
        stack.extend(v for v in vars(node).values() if isinstance(v, ParseNode))
    return total


def nested_g(n):
    result = Var('p')
    for _ in range(n): result = Modality(['a'], Globally(result))
    return result


def closure_copy_work(formula):
    # Instrument top-level deepcopy calls only; timing runs remain untouched.
    original = builder.deepcopy
    copied = []
    def measured(node):
        copied.append(node_count(node))
        return original(node)
    try:
        builder.deepcopy = measured
        builder.generate_closure(formula)
    finally:
        builder.deepcopy = original
    return dict(copy_calls=len(copied), copied_ast_nodes=sum(copied))


def timed(call):
    start = perf_counter(); value = call()
    return value, perf_counter() - start


def run(repeats):
    rows = []
    g = CGS(); g.add_agent('a'); g.add_decisions('a', {'stay'})
    g.add_proposition('p'); g.set_initial_state('s'); g.label_state('s', {'p'})
    g.add_transition('s', [('a','stay')], 's')
    for n in (8,16,32,64):
        raw = nested_g(n)
        work = closure_copy_work(normalize_formula(raw))
        if work['copied_ast_nodes'] != (n+1)**2:
            raise AssertionError('Copy-work formula no longer matches implementation; revise this audit.')
        for repetition in range(repeats):
            gc.collect()
            f, norm_time = timed(lambda: normalize_formula(raw))
            a, build_time = timed(lambda: build_acg_final(f,g))
            rows.append(dict(family='nested_g_fixed_model',parameter=n,repeat=repetition,
                ast_nodes=node_count(f),acg_states=len(a.states),
                normalization_seconds=norm_time,acg_seconds=build_time,**work))
    for n in (1,2,3):
        raw=generate_flatG_spec(n)
        for repetition in range(repeats):
            gc.collect()
            g, cgstime=timed(lambda:generate_lights_cgs(n))
            f, normtime=timed(lambda:normalize_formula(raw))
            a, acgtime=timed(lambda:build_acg_final(f,g))
            arena, arenatime=timed(lambda:build_game(a,g))
            regions, solvertime=timed(lambda:solve_buchi_game(*arena[:5]))
            accepted=arena[5] in regions[0]
            # All p_i start false, so each positive G obligation fails now.
            if accepted: raise AssertionError('Unexpected result on flat positive G family.')
            rows.append(dict(family='lights_flat_g',parameter=n,repeat=repetition,
                ast_nodes=node_count(f),acg_states=len(a.states),agents=len(g.agents),
                cgs_states=len(g.states),cgs_transitions=len(g.transition_function),
                arena_vertices=len(arena[0]),arena_edges=len(arena[1]),satisfied=accepted,
                cgs_seconds=cgstime,normalization_seconds=normtime,acg_seconds=acgtime,
                arena_seconds=arenatime,solver_seconds=solvertime))
    files=sorted(p for folder in ('preprocessing','acg','acceptance_game','buchi_solver','benchmarks')
                 for p in Path(folder).rglob('*.py'))
    return dict(metadata=dict(python=platform.python_version(),system=platform.system(),
        machine=platform.machine(),repeats=repeats,production_parent='2e2c9c777ce2beda53c9c9c819099d3b42414b01',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        notes=['Formula generation and result printing excluded from stage timings.',
               'Arena time includes CGS validation; constructor uses symbolic alphabet.',
               'Reachable arena plus sinks; no parser timing (input is an AST).',
               'Copy instrumentation is a separate untimed run.',
               'Tiny diagnostic sample; no asymptotic fit or historical reproduction.']),rows=rows)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1: parser.error('repeats must be positive')
    result=run(args.repeats)
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(f"Saved {len(result['rows'])} measurements to {args.output}")
