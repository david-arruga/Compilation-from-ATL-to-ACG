"""Compare constructor-only timings on already-normalized nested G inputs."""
import gc
import json
import platform
from statistics import median
from time import perf_counter
from acg import build_acg_final, build_acg_indexed
from scripts.performance_audit import nested_g
from acg import CGS


def run():
    g=CGS();g.add_agent('a')
    rows=[]
    for n in (8,16,32,64,256,1024,4096):
        f=nested_g(n)
        constructors=[('indexed',build_acg_indexed)]
        if n<=64:constructors.insert(0,('reference',build_acg_final))
        for name,constructor in constructors:
            times=[]
            for _ in range(3):
                gc.collect()
                start=perf_counter();a=constructor(f,g);times.append(perf_counter()-start)
            rows.append(dict(depth=n,ast_nodes=2*n+1,compiler=name,states=len(a.states),
                             seconds=times,median_seconds=median(times)))
    return dict(python=platform.python_version(),system=platform.system(),
                scope='Already-normalized AST to symbolic ACG; fixed single agent; no rendering, normalization, game or solver.',
                rows=rows)


if __name__=='__main__': print(json.dumps(run(),indent=2))
