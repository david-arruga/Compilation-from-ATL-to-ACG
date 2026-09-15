"""Separate GC pause measurements from cProfile call counts (not a time proof).

Run from repository root: PYTHONHASHSEED=0 python3 -m scripts.profile_indexed
"""
import cProfile
import gc
import hashlib
import json
import platform
import pstats
import random
from pathlib import Path
from time import perf_counter

from acg import CGS, build_acg_indexed
from scripts.performance_audit import nested_g


def run():
    model = CGS()
    model.add_agent('a')
    inputs = {d: nested_g(d) for d in (4096, 8192, 16384, 32768)}
    rows = []
    order = [(d, enabled, rep) for d in inputs for enabled in (True, False)
             for rep in range(5)]
    random.Random(20260916).shuffle(order)
    previous = gc.isenabled()
    try:
        for depth, enabled, rep in order:
            gc.collect()
            pauses, starts = [], {}

            def callback(phase, info):
                generation = info['generation']
                if phase == 'start':
                    starts[generation] = perf_counter()
                else:
                    pauses.append(dict(generation=generation,
                        seconds=perf_counter()-starts.pop(generation),
                        collected=info['collected']))

            gc.callbacks.append(callback)
            (gc.enable if enabled else gc.disable)()
            try:
                start = perf_counter()
                out = build_acg_indexed(inputs[depth], model)
                elapsed = perf_counter()-start
            finally:
                gc.callbacks.remove(callback)
            rows.append(dict(nodes=2*depth+1, gc_enabled=enabled, repeat=rep,
                             seconds=elapsed, gc_seconds=sum(p['seconds'] for p in pauses),
                             collections_by_generation={str(g): dict(count=sum(p['generation']==g for p in pauses),
                                 seconds=sum(p['seconds'] for p in pauses if p['generation']==g),
                                 collected=sum(p['collected'] for p in pauses if p['generation']==g))
                                 for g in range(3)}))
            del out
        profiles = []
        # Profiling adds overhead: use these runs for attribution and counts,
        # never compare their times to the unprofiled scaling benchmark.
        gc.disable()
        for depth in inputs:
            gc.collect()
            profiler = cProfile.Profile()
            out = profiler.runcall(build_acg_indexed, inputs[depth], model)
            stats = pstats.Stats(profiler)
            functions = []
            for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
                functions.append(dict(file=Path(filename).name, line=line, name=name,
                                      calls=calls, primitive_calls=primitive,
                                      own_seconds=own, cumulative_seconds=cumulative))
            profiles.append(dict(nodes=2*depth+1, total_calls=stats.total_calls,
                                 functions=sorted(functions, key=lambda f:-f['own_seconds'])))
            del out
    finally:
        (gc.enable if previous else gc.disable)()
    return dict(python=platform.python_version(),
                compiler_sha256=hashlib.sha256(Path('acg/indexed.py').read_bytes()).hexdigest(),
                methodology='Fixed one-agent nested G family; randomized interleaved GC modes; '
                '5 repeats per size/mode. GC callback overhead included; cProfile runs separate. '
                'Input creation, explicit pre-run collection and output destruction excluded.',
                measurements=rows, profiles=profiles)


if __name__ == '__main__':
    Path('docs/indexed_profile.json').write_text(json.dumps(run(), indent=2)+'\n')
