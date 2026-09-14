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
