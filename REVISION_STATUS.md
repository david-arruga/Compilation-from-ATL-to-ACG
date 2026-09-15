# Revisión contra MAIN_REVISADO v0.26

Rama corregida: `codex/thesis-alignment`.
Base histórica: `849c34c4d131d06b4bf49512038c3d2d45a83b21` en `main`.
La versión histórica permanece intacta. No se duplican archivos con sufijo `_new`.

## Bloque 1 — entrada y contrato del constructor

- `preprocessing/validator.py`: nueva validación de la gramática normalizada que
  realmente implementa el constructor. Comprueba coaliciones cuando se conoce
  el universo de agentes; no permite eludir la gramática mediante la marca interna
  de `Eventually`. El filtro estricto devuelve `UNSUPPORTED` para lo que queda
  fuera de este contrato; no afirma que esas fórmulas sean semánticamente inválidas.
- `preprocessing/transformer.py`: la normalización aplica las dualidades soportadas
  y comprueba su resultado. Una negación de camino no se transforma indebidamente
  en negación de cuantificador estratégico. Release positivo y dual Until siguen
  fuera del contrato: se rechazan explícitamente, no se declara una nueva traducción.
- `acg/builder.py`: exige el contrato y coaliciones contenidas en los agentes.
  Conserva las seis reglas estratégicas y la aceptación anteriores.
- `ACG/` se renombra a `acg/` para coincidir con los imports en Linux.
  `__init__.py`, `cgs.py` y `model.py` solo cambian de ruta en este bloque.
- `main.py`: valida el CGS seleccionado y denomina el resultado satisfacción en
  ese modelo, no satisfacibilidad general.
- `tests/test_compiler_contract.py`: regresiones y comprobaciones semánticas.

## Verificación realizada

`python3 -m unittest discover -s tests -v`: 8 pruebas correctas.
Incluye tabla de verdad booleana, rechazo de entradas no soportadas, ejemplo
PowerPlant de 10 estados, totalidad de transiciones para las seis formas
estratégicas y 192 comparaciones semánticas de Next positivo, negativo y dual
sobre los 16 juegos booleanos de dos agentes y las cuatro coaliciones.
No se ejecuta el solver histórico como oráculo de corrección.
Estas pruebas no equivalen a una verificación formal universal del Python.

## Pendientes; no están aprobados por este bloque

1. Parser: correcciones y regresiones del bloque 2; futuras ampliaciones de sintaxis requieren pruebas.
2. G/U: comprobación exhaustiva en modelos deterministas de dos estados (bloque 2); falta cubrir elecciones estratégicas concurrentes y auditoría general.
3. Generación del juego y totalización, cobertura de soportes, validación de CGS.
4. Solver: corrección de subarenas realizada y contrastada en bloque 3; pendiente análisis del coste de esta implementación.
5. Coste Python: copias, hashing recursivo, normalización y estructuras de datos.
   La cota abstracta Lean no certifica el tiempo de este programa.
6. Reproducir experimentos y alinear capítulos 4 y 5 con una versión identificada.
7. README histórico: rutas y ejemplos adicionales necesitan actualización.

No se ha cambiado MAIN_REVISADO ni se han revalidado los resultados experimentales.

## Bloque 2 — parser y regresiones temporales

Archivos corregidos: `preprocessing/parser.py`.
Pruebas añadidas: `tests/test_parser_temporal.py`.

Fallos reproducidos antes del cambio:
- `Fuel` se interpretaba como F aplicado a `uel`.
- Identificadores `ctrl_0` y `p_0` no se podían leer.
- `(<a> X p) U q` movía indebidamente el cuantificador dentro del Until.
- Se aceptaban coaliciones mal separadas como `<a,>` o `<a b>`.
- No se leían constantes ni varios símbolos emitidos por `to_formula()`.

El lexer ahora lee identificadores completos, permite guiones bajos, constantes
`true`, `false`, `⊤`, `⊥`, conectivas Unicode y flechas ASCII.
Las abreviaturas temporales X/G/F/U/R se reconocen solo como tokens completos.
La precedencia queda documentada: unarios, U/R, and, or, implies, iff.
U/R e implicación asocian a la derecha; iff a la izquierda.
Un operando booleano de X/G/F requiere paréntesis. Se conserva `<a>p U q`
y se recomienda la forma inequívoca `<a>(p U q)`.
El parser puede representar formas que la validación posterior rechaza;
aceptarlas sintácticamente no amplía el fragmento compilado.

Validación: 14 pruebas unittest correctas. Las nuevas pruebas incluyen 1024
comparaciones del autómata y arena generados frente a la semántica directa
sobre la trayectoria única, para todos los mapas de transición deterministas
de dos estados, todas sus etiquetas p/q y ambos estados iniciales.
Cubren G, U, F, un G con F estratégico anidado y sus negaciones.
La aceptación se calcula con un oráculo de puntos fijos independiente del
solver histórico; no certifica el solver de producción. Los modelos de esta
familia tienen una acción por agente, por lo que no cubren elecciones estratégicas
concurrentes para G/U. Permanecen las 192 comparaciones estratégicas de Next.
Las seis reglas del constructor no han necesitado cambios en este bloque.

## Bloque 3 — solver y elecciones concurrentes G/U

Archivos corregidos: `buchi_solver/solver.py`, `buchi_solver/predecessor.py`.
Pruebas añadidas: `tests/test_buchi_concurrent.py`.

Contraejemplo del solver anterior (reproducido antes del cambio):
V={0,1,2,3}; Accept={2,3}; Reject={0,1}; B={1,3}.
Aristas: 0→0, 1→0, 2→1, 2→2, 3→0, 3→2.
Todas las posiciones son perdedoras para Accept: 1 y 3 no pueden visitarse
infinitamente. El solver antiguo devolvía {3} como ganadora, porque permitía
atractores que pasaban por regiones eliminadas. El solver corregido devuelve
la región vacía. Ahora los atractores se calculan con aristas y propietarios
restringidos al conjunto restante, como en alg:solve-buchi de MAIN_REVISADO.

La API comprueba partición de propietarios, pertenencia de aristas y B, y
la ausencia de terminales sin sucesor. La arena vacía es válida. Se elimina
una impresión de depuración en predecessor_1. No se cambia la definición
matemática de predecesor ni se introduce un nuevo algoritmo teórico.

Validación: 18 pruebas correctas (`python3 -m unittest discover -s tests -v`).
- Las 21.952 arenas totales de tres vértices (todas las aristas, propietarios
  y conjuntos B posibles) coinciden con el oráculo independiente de puntos
  fijos anidados. Ese tamaño por sí solo no detectaba el fallo: se conserva
  por separado el contraejemplo de cuatro vértices.
- 2.048 comparaciones extremo a extremo para G/U y negaciones sobre 64 CGS
  de dos estados seleccionados con semilla 20260914; dos agentes, dos acciones
  por agente, todas las coaliciones y ambos estados iniciales. Se compara
  con la semántica ATL por Pre y puntos fijos calculada directamente en el
  CGS, sin utilizar el ACG para construir el resultado esperado.
- La muestra no es exhaustiva sobre todos los CGS y no equivale a verificar
  formalmente Python. Permanecen las pruebas anteriores.

Ejecución del ejemplo del capítulo 4:
`python3 main.py --formula '<Valve> globally (underpowered implies <Reactor> next efficient)' --cgs 1`
Resultado: satisfacción YES, 10 estados ACG, arena de 77 vértices y 109 aristas.
El contador histórico ACG size=20 suma estados y átomos distintos; no son
20 estados ni una medición completa del espacio ocupado en Python.

Pendientes: auditoría general de arena/CGS y soportes (incluidas entradas
ACG externas al compilador), coste y representación, reproducción de datos y
actualización de capítulos 4/5. No se ha modificado LaTeX ni certificado los
experimentos anteriores. Las pruebas de Lean no se atribuyen a este código.

## Bloque 4 — contrato CGS y soportes de transición

Corregidos: `acg/cgs.py`, `acceptance_game/builder.py`,
`acceptance_game/expansion.py`, `acceptance_game/utils.py`,
`acceptance_game/examples.py`. Pruebas: `tests/test_arena_contract.py`.

- Un conjunto de movimientos ausente se rechaza con ValueError, no KeyError.
- Se comprueban extremos de transiciones, asignación de una acción legal por
  agente, etiquetas y movimientos no vacíos. No se exige alcanzabilidad salvo
  petición explícita. None permanece reservado en esta representación.
- Una transición contradictoria no sobrescribe la anterior; una transición
  inexistente no desaparece silenciosamente al consultar su sucesor.
- build_game valida el CGS, estado inicial y aceptación ACG. Las transiciones
  visitadas se comprueban antes de expandirlas (tipos, estados destino, agentes).
  Esto no inspecciona todas las transiciones inalcanzables de un ACG externo.
- Top tiene soporte vacío; Bottom no tiene soportes. Las constantes anidadas
  ahora se expanden correctamente. El soporte vacío lleva al sumidero ganador;
  la ausencia de soportes al perdedor, evitando vértices sin sucesores.
- cgs2 declaraba wait mientras todas las transiciones usaban dontWalk; la
  declaración ahora sigue la tabla existente. Se declaran peds_cross, peds_wait
  y emergency_peds, ya presentes en etiquetas. No se cambian destinos ni etiquetas.
  Este arreglo estructural no revalida la interpretación ni los experimentos
  históricos del ejemplo. Los cuatro modelos ahora superan validate().

Validación: 25 pruebas correctas. Se añaden 1152 comparaciones de soportes con
asignaciones booleanas, casos de constantes anidadas, rechazo de modelos
malformados, conservación de estados inalcanzables y validación de los ejemplos.
Se conservan las pruebas concurrentes y del solver. No es una prueba universal
ni una verificación Lean del Python. Las acciones siguen siendo globales por
agente, especialización de los movimientos locales de la tesis.

Pendientes principales: interpretación detallada de ejemplos/capítulos 4 y 5,
reproducibilidad experimental y coste real del constructor y del solver.

## Bloque 5 — reproducción del ejemplo del capítulo 4

Añadidos `scripts/chapter4_trace.py` y `docs/CHAPTER4_WALKTHROUGH.md`.
La traza verifica la normalización, los 10 estados y aceptación ACG, todas sus
transiciones sobre las cuatro valoraciones, las 20 transiciones PowerPlant y
los cinco estados iniciales posibles. Resultado: g vale en s0 y s4; falla en
s1, s2, s3. Desde s0: 77 vértices, 109 aristas, 23 ganadores y 54 perdedores.
Ejecutado `python3 -m scripts.chapter4_trace` correctamente. El código de
producción no cambia en este bloque; no se repiten los tests ya superados.

Diferencias documentadas: núcleo de 10 estados frente a clausura de 12;
alfabeto simbólico, literales evaluados bajo demanda y contador estados+átomos;
arena alcanzable con nodos compartidos frente a pseudocódigo completo con
identidades más detalladas; soportes no necesariamente mínimos. Este cotejo
no acredita una equivalencia universal de esas representaciones. No se ha
modificado MAIN_REVISADO ni revalidado los experimentos históricos.


## Bloque 6 — presentación del juego alineada con el texto

Corregidos `acceptance_game/expansion.py`, `utils.py`, `builder.py` y `main.py`.
Los vértices estratégicos conservan ahora origen y soporte, además de átomo y
movimiento. La generación devuelve todos los soportes mínimos. La opción
`build_game(..., full_arena=True)` / `--full-arena` construye Q×S completo.
La alternativa por defecto explora la raíz y conserva ambos sumideros.

El ejemplo mantiene 77 vértices/109 aristas en modo alcanzable; en modo completo
produce 152/216 (Q core=10, S=5). Se añade en la guía el argumento de preservación
por restricción cerrada a sucesores y la justificación de minimización de soportes.
No se añade un teorema Lean ni se afirma coste lineal de enumerar soportes.
Las notas de discrepancias de bloques 5 y anteriores describen esas versiones:
las identidades compartidas y soportes no mínimos ya no se usan en esta rama.

Validación: las 25 pruebas anteriores siguen pasando; tres nuevas pruebas pasan
(28 en total): soportes mínimos, identidades distintas para fuentes/soportes,
y comparación exacta de la arena alcanzable con la restricción de la completa
para tres fórmulas, incluyendo propietarios, aceptación y regiones ganadoras.
La traza del capítulo 4 también pasa. MAIN_REVISADO y PDF histórico intactos.

## Bloque 7 — diagnóstico de coste y evidencia experimental

Añadidos `scripts/performance_audit.py`, `docs/performance_sample.json` y
`docs/PERFORMANCE_AUDIT.md`. No cambia código de producción.
Ejecutado el diagnóstico: 21 mediciones, con tiempos separados y hashes fuente.

Hallazgo analítico: para φ_n=⟨a⟩G repetido n veces sobre p, el AST tiene 2n+1
nodos pero las llamadas a deepcopy de la clausura copian (n+1)² nodos en total.
Por tanto este Python no cumple coste lineal de peor caso bajo conteo de visitas
AST. No contradice la construcción lineal formalizada con otra representación.
Medianas ACG n=8,16,32,64: 0.828, 3.084, 13.388, 55.351 ms; estos tiempos son
observaciones locales, no la demostración de la cota inferior.

En lights, n produce 2^n estados y 4^n transiciones CGS. No confundir n con el
tamaño explícito del modelo. Muestra pequeña n=1..3, 3 repeticiones por tamaño,
con resultado esperado controlado; no reproduce figuras ni comparadores antiguos.
No se localizaron datos brutos ni scripts de figuras históricas en el repositorio.

Siguiente frente: constructor indexado sin copias/hashing de árboles completos,
con correspondencia semántica explícita. Mantener como referencia el constructor
actual y sus comprobaciones. No se actualiza LaTeX ni se proclama rendimiento
lineal del software a partir de una curva o del número de estados.

## Bloque 8 — constructor indexado

Añadido `acg/indexed.py`: recorrido iterativo por ocurrencias y polaridad,
transiciones sobre IDs, sin copia/hashing de subárboles. Se exporta desde acg.
`main.py` utiliza el constructor indexado por defecto y conserva
`--compiler reference`. Los imports explícitos build_acg_final no cambian.
Documentación en `docs/INDEXED_COMPILER.md`; comparador reproducible en
`scripts/compare_compilers.py`, con datos en `docs/indexed_performance_sample.json`.

Validación: 32 pruebas correctas; 82 entradas cotejadas por proyección completa
de transiciones y aceptación frente a la referencia. Compilación de 5000 G
sin recursión de Python, 10001 visitas AST y 10002 estados. Ejemplo CLI con
indexado: YES, 77 vértices, 109 aristas. Constructor de referencia intacto.

La linealidad se argumenta en el modelo de referencias/contenedores con agentes
fijos; no se afirma una verificación del tiempo de CPython ni la constante exacta
Lean. Normalización/renderizado/alfabeto explícito/juego/solver quedan separados.
Pendientes: migración explícita de benchmarks y documentación de ejemplos,
reproducción de evaluación y traslado de estas notas al LaTeX revisado.


## Bloque 9 — gráfica de escalado indexado

Añadidos script scripts/plot_indexed_scaling.py, informe docs/INDEXED_SCALING.md,
datos indexed_scaling.json / indexed_scaling_gc_disabled.json y gráfica PNG/SVG.
180 mediciones de constructor sobre AST normalizado; 10 tamaños × 9 repeticiones
× 2 modalidades de GC. Visitas y tamaños estructurales comprobados en cada medida.

Resultado matizado: tiempo/nodo aproximadamente estable hasta 8193 nodos;
desviación en los mayores, parcialmente reducida desactivando GC durante la
construcción. 32769 nodos: 298.603 ms con GC, 144.967 ms sin GC (medianas locales).
No se afirma una recta perfecta ni una prueba asintótica por regresión.
No se modifica producción, Lean, LaTeX ni la evaluación histórica.
