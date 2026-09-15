"""Measure normalized AST -> indexed symbolic ACG and plot observed scaling.

Run from root: PYTHONHASHSEED=0 python3 -m scripts.plot_indexed_scaling
Matplotlib is needed for plotting only. Production code is not instrumented.
"""
import gc
import hashlib
import json
import os
import platform
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from acg import CGS, build_acg_indexed
from scripts.performance_audit import nested_g

SIZES = [32,64,128,256,512,1024,2048,4096,8192,16384]
REPEATS = 9


def measure(gc_during=True):
    g=CGS();g.add_agent('a')
    # Formulas already satisfy the normalized grammar; their construction
    # is outside all timed regions. No parser/normalizer/game/printing here.
    inputs={n:nested_g(n) for n in SIZES}
    warm=build_acg_indexed(inputs[32],g)
    del warm
    rows=[]
    order=[(n,r) for r in range(REPEATS) for n in SIZES]
    random.Random(20260915).shuffle(order)
    for n,r in order:
        gc.collect()  # outside timer; automatic GC remains enabled inside it
        if not gc_during: gc.disable()
        try:
            start=perf_counter()
            automaton=build_acg_indexed(inputs[n],g)
            elapsed=perf_counter()-start
        finally:
            gc.enable()
        rows.append(dict(depth=n,ast_nodes=2*n+1,repeat=r,seconds=elapsed,
                         visits=automaton.input_visits,states=len(automaton.states),
                         stored_transition_schemas=len(automaton.transitions)))
        assert automaton.input_visits==2*n+1
        assert len(automaton.states)==2*n+2
        assert len(automaton.transitions)==2*n
        del automaton  # destruction is outside timing
    summaries=[]
    for n in SIZES:
        times=[x['seconds'] for x in rows if x['depth']==n]
        q1,_,q3=statistics.quantiles(times,n=4,method='inclusive')
        med=statistics.median(times)
        summaries.append(dict(depth=n,ast_nodes=2*n+1,median_seconds=med,
                              q1_seconds=q1,q3_seconds=q3,us_per_node=med*1e6/(2*n+1)))
    source=Path('acg/indexed.py')
    return dict(metadata=dict(utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),system=platform.system(),machine=platform.machine(),
        compiler_commit='7769610ceae2a165f54dc6cf10f3e72de900586e',
        indexed_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        hash_seed=os.environ.get('PYTHONHASHSEED'),gc_enabled_during_construction=gc_during,
        repeats=REPEATS,order_seed=20260915,fixed_agents=1,fixed_propositions=1,
        scope='Already-normalized AST to complete indexed symbolic ACG, including validation; excludes input creation, normalizer, rendering, CGS/game/solver, and output destruction.',
        warning='Finite sample on one formula family and runtime; not an asymptotic proof.'),
        measurements=rows,summary=summaries)


def plot(data, control):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    rows=data['summary']
    x=[r['ast_nodes'] for r in rows]
    y=[1000*r['median_seconds'] for r in rows]
    lo=[1000*r['q1_seconds'] for r in rows];hi=[1000*r['q3_seconds'] for r in rows]
    fig,axs=plt.subplots(1,2,figsize=(12,4.9),layout='constrained')
    fig.suptitle('Constructor ACG indexado: escalado observado',fontsize=16,fontweight='bold')
    ax=axs[0]
    ax.plot(x,y,'o-',color='#16697a',label='GC automático: mediana')
    ax.fill_between(x,lo,hi,color='#16697a',alpha=.18,label='Cuartiles 25–75 %')
    # Reference uses the median time-per-node of the first 5 sizes;
    # it is an illustrative slope, not an upper bound or fitted theorem.
    slope=statistics.median(y[i]/x[i] for i in range(5))
    ax.plot([0,x[-1]],[0,slope*x[-1]],'--',color='#d68130',label='Referencia proporcional a N¹')
    ax.set(xlabel='N: nodos de la fórmula normalizada',ylabel='Tiempo de construcción (ms)',xlim=(0,None),ylim=(0,None))
    axs[1].plot(x,[r['us_per_node'] for r in rows],'o-',color='#16697a')
    axs[1].fill_between(x,[1000*l/n for l,n in zip(lo,x)],
                        [1000*h/n for h,n in zip(hi,x)],color='#16697a',alpha=.18)
    axs[1].set(xlabel='N: nodos de la fórmula normalizada',ylabel='Tiempo por nodo (µs)',ylim=(0,None))
    control_rows=control['summary']
    cx=[r['ast_nodes'] for r in control_rows]
    cy=[r['median_seconds']*1000 for r in control_rows]
    axs[0].plot(cx,cy,'s-',color='#8a397b',label='Sin GC automático: mediana')
    axs[1].plot(cx,[r['us_per_node'] for r in control_rows],'s-',color='#8a397b')
    axs[1].set_xscale('log',base=2)
    for a in axs:
        a.grid(alpha=.2);a.spines[['top','right']].set_visible(False)
        a.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{int(v):,}'.replace(',',' ')))
    ax.legend(fontsize=8,loc='upper left')
    fig.supxlabel('Un agente y una proposición fijos · G anidados · 9 repeticiones por tamaño y modalidad\n¹Pendiente de referencia: mediana de T/N en los cinco tamaños menores. No es una cota.',fontsize=9)
    fig.savefig('docs/indexed_scaling.png',dpi=170)
    fig.savefig('docs/indexed_scaling.svg')
    plt.close(fig)


if __name__=='__main__':
    data=measure()
    Path('docs/indexed_scaling.json').write_text(json.dumps(data,indent=2)+'\n')
    control=measure(gc_during=False)
    Path('docs/indexed_scaling_gc_disabled.json').write_text(json.dumps(control,indent=2)+'\n')
    plot(data,control)
    for row in data['summary']:
        other=next(x for x in control['summary'] if x['ast_nodes']==row['ast_nodes'])
        print(row['ast_nodes'],round(row['median_seconds']*1000,3),round(other['median_seconds']*1000,3))
