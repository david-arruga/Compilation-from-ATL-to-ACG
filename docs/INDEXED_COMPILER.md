# Constructor ACG por índices

`acg/indexed.py` añade `build_acg_indexed`. La CLI lo utiliza por defecto;
`--compiler reference` selecciona el constructor anterior, conservado sin cambios.
Ambos reciben fórmulas ya normalizadas. La normalización y el parser siguen
siendo fases distintas y no se han optimizado en este bloque.

## Representación

Cada ocurrencia de fórmula de estado, salvo el envoltorio Not, recibe dos IDs:
2i y 2i+1, que representan la fórmula y su negación normalizada. Not cambia la
polaridad del resultado de su hijo. Los registros se emiten en postorden;
contienen tipo, dato atómico/coalición y como máximo dos referencias a hijos.
Las ocurrencias repetidas no se deduplican mediante igualdad de fórmulas.

Las transiciones referencian IDs; G positivo y U negativo conservan la aceptación
Büchi. Literales se evalúan bajo demanda. El constructor no ejecuta deepcopy,
no utiliza AST como clave de diccionario o conjunto y no renderiza fórmulas.
El recorrido es iterativo, por lo que compilar un AST profundo no depende de
la pila recursiva del intérprete. La entrada debe ser un AST finito y acíclico.

`decode_labels()` reconstruye etiquetas para cotejo y explicación; queda fuera
de la compilación y del juego. Ordena las coaliciones para imprimirlas: el orden
y las repeticiones de agentes en la sintaxis no cambian la coalición matemática.
Las comparaciones/impresión posterior de estas etiquetas pueden ser costosas.

## Correspondencia matemática

Definimos L(2i) como la fórmula del registro usando las etiquetas de sus hijos,
y L(2i+1) como su negación normalizada. El orden postfijo hace esta definición
bien fundada. Cada regla emitida coincide bajo L con la regla de la tesis;
la autocontinuación estratégica apunta al propio ID con su polaridad.
El estado inicial tiene etiqueta igual a la entrada normalizada (coaliciones
entendidas como conjuntos), y la aceptación coincide con G positivo/U negativo.

Así, un run indexado se proyecta aplicando L a sus estados. Para levantar un
run de fórmulas desde una etiqueta L(q), en cada nodo se usan los hijos indexados
prescritos por la transición de q y se continúa con el mismo procedimiento;
no se necesita seleccionar una única ocurrencia global para cada fórmula.
En ambos sentidos se conservan las obligaciones, movimientos y aceptación.
Este argumento es la correspondencia de representación que justifica el diseño;
no es una nueva verificación Lean del código Python ni identidad de implementaciones.

## Coste declarado y límites

Para N ocurrencias AST se hacen N visitas y como máximo 2N estados. Cada registro
emite un número constante de nodos de transición. No hay recorridos repetidos de
subárboles. Con agentes fijos y operaciones sobre referencias/contenedores de
coste unitario habitual, la construcción simbólica realiza trabajo lineal en N.
Con a agentes variables, el tratamiento/complemento de coaliciones da la cota
O(N(1+a)), además del tratamiento de los nombres de entrada. No se afirma el
coeficiente exacto (12+4k) de Lean, ni tiempo de máquina formalmente verificado.

Los diccionarios/conjuntos Python tienen supuestos de hashing y costes amortizados;
los IDs enteros también tienen costes de representación. Se explicitan estos
supuestos en vez de prometer una cota bit a bit de CPython. Materializar 2^AP,
reconstruir/imprimir etiquetas, normalizar, construir el CGS, el juego y resolverlo
no forman parte de esta cota del constructor simbólico.

## Comprobaciones y medición

La suite suma 32 pruebas correctas. Las nuevas cotejan 82 entradas normalizadas:
raíz, imagen de etiquetas, aceptación, proyección de cada transición para cuatro
valoraciones y resultado del juego frente al constructor de referencia.
Se rechazan entradas fuera del contrato; el ejemplo PowerPlant satisface la
fórmula con ambos compiladores. Una prueba compila 5000 G anidados con copia y
hashing base de AST bloqueados, comprobando 10001 visitas y 10002 estados.
Las pruebas finitas no constituyen una certificación universal del Python.

```bash
python3 -m unittest discover -s tests -v
PYTHONHASHSEED=0 python3 -m scripts.compare_compilers
python3 main.py --compiler indexed --formula '<Valve> globally (underpowered implies <Reactor> next efficient)' --cgs 1
```

Datos nuevos en `indexed_performance_sample.json`, tres repeticiones por caso.
En esta ejecución, n=64 pasó de una mediana de 175.023 ms (referencia) a
0.395 ms (indexado). Para n=256/1024/4096 el indexado dio 1.710/6.910/54.726 ms.
Son observaciones locales con variación y GC, no una demostración por ajuste.
No comparar estos números con sesiones antiguas como si el entorno fuera idéntico.

El constructor anterior sigue siendo útil como referencia legible. Al usar
índices, las salidas imprimen q0, q1, etc.; decode_labels permite recuperar su
significado. Ejemplos y benchmarks que importan explícitamente build_acg_final
siguen utilizando el constructor anterior: la migración no se ha hecho en silencio.
