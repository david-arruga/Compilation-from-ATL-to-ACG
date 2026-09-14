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

1. Auditoría completa del parser (tokens, precedencia y cobertura de sintaxis).
2. Auditoría semántica restante de normalización y constructor; pruebas de G/U.
3. Generación del juego y totalización, cobertura de soportes, validación de CGS.
4. Solver: atractores sobre la subarena restante, no sobre la arena original.
5. Coste Python: copias, hashing recursivo, normalización y estructuras de datos.
   La cota abstracta Lean no certifica el tiempo de este programa.
6. Reproducir experimentos y alinear capítulos 4 y 5 con una versión identificada.
7. README histórico: rutas y ejemplos adicionales necesitan actualización.

No se ha cambiado MAIN_REVISADO ni se han revalidado los resultados experimentales.
