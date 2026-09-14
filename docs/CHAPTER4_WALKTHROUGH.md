# Ejemplo del capítulo 4: del texto al programa

Referencia matemática: MAIN_REVISADO, revisión interna v0.26, tablas
`tab:SABROSA`, `tab:SABROSA2`, `fig:Automata exodia` y el apartado
«The power-plant example: a move versus a strategy».
Código de partida de este cotejo: `3ac1a1eca6d4cd251fe2da5b1cee73207c8a18fe`.
Esta guía corresponde a la rama corregida, no a los experimentos históricos.

## Reproducción

Desde la raíz del repositorio, sin dependencias adicionales para esta ejecución:

```bash
python3 -m scripts.chapter4_trace
```

El script comprueba la tabla transcrita del texto, las 20 transiciones del CGS,
las transiciones de los 10 estados ACG para las cuatro valoraciones, y la
satisfacción desde los cinco estados. Imprime JSON determinista con los resultados.
Se debe ejecutar sin `-O`, pues las comprobaciones usan `assert`.

Para usar la interfaz habitual:

```bash
python3 main.py --formula '<Valve> globally (underpowered implies <Reactor> next efficient)' --cgs 1
```

## Qué se está preguntando

¿Puede Valve garantizar que, siempre que la planta esté underpowered,
Reactor pueda garantizar efficient en el siguiente paso?

Sea u=underpowered, e=efficient, x=⟨Reactor⟩Xe,
b=¬u∨x y g=⟨Valve⟩Gb. La fórmula normalizada es g.
La negación normalizada de b es **u∧¬x**, no u∨¬x.

| Código CGS | Nombre del texto |
|---|---|
| s0 | start |
| s1 | efficient |
| s2 | underpowered |
| s3 | danger |
| s4 | shutdown |

La tabla de transiciones del texto coincide con el código en sus 20 entradas.
Los rótulos gráficos antiguos que el texto ya señala como inconsistentes no
se utilizan como referencia para modificar esa tabla.

## Autómata

Estados: g, ¬g, b, ¬b, x, ¬x, u, ¬u, e, ¬e. Inicial: g. Büchi: {g}.
El alfabeto matemático tiene cuatro valoraciones: ∅, {u}, {e}, {u,e}.
`Alphabet: []` en la impresión histórica indica que el alfabeto no se ha
materializado; no significa que el alfabeto matemático sea vacío.

| Estado | Transición |
|---|---|
| g | (b,ε) ∧ (g,□,{Valve}) |
| ¬g | (¬b,ε) ∨ (¬g,◇,{Reactor}) |
| b | (¬u,ε) ∨ (x,ε) |
| ¬b | (u,ε) ∧ (¬x,ε) |
| x | (e,□,{Reactor}) |
| ¬x | (¬e,◇,{Valve}) |
| u, e | Verdadero exactamente cuando la proposición está en la etiqueta |
| ¬u, ¬e | Verdadero exactamente cuando la proposición no está en la etiqueta |

Las transiciones literales se evalúan al consultarlas y no aparecen como
entradas almacenadas en la impresión antigua del diccionario de transiciones.
La traza sí comprueba sus cuatro valoraciones.

La clausura histórica completa del texto contiene 12 estados; estos 10 son
el núcleo descrito expresamente en el capítulo 4. Desde g, los únicos estados
ACG alcanzables por obligaciones son g, b, ¬u, x y e. Los negativos restantes
siguen siendo parte del autómata y se comprueban también.

El antiguo indicador `ACG size: 20` significa 10 estados más 10 átomos distintos.
No significa 20 estados, bytes de memoria ni un coste temporal medido.

## Resultado y explicación estratégica

La arena que produce el programa desde start tiene 77 vértices, 109 aristas,
36 vértices de Accept, 41 de Reject y 6 aceptantes. El solver obtiene 23
vértices ganadores de Accept y 54 perdedores. Son vértices del juego, no estados
CGS ni estados ACG. La posición inicial pertenece a la región ganadora.

| Estado de evaluación | ¿Satisface g? |
|---|---|
| start | Sí |
| efficient | No |
| underpowered | No |
| danger | No |
| shutdown | Sí |

Desde start, Valve puede elegir siempre open: tanto heat como cool dejan la
planta en start. Allí u es falso. Si Reject comprueba la obligación b, Accept
elige ¬u y gana; si sigue la continuación indefinidamente, visita g infinitamente.
La satisfacción procede de evitar underpowered, no de demostrar recuperación.

En underpowered, Reactor no puede forzar efficient: heat puede ser contestado
por open, y cool lleva a start. Así, b ya es falso allí. Desde efficient,
Reactor puede elegir cool y llevar a underpowered cualquiera que sea la acción
de Valve. Desde danger, cool lleva a underpowered o a efficient. Shutdown, en
cambio, satisface ¬u y todas sus transiciones llevan a start.

## Diferencias que no deben ocultarse

El pseudocódigo del capítulo genera configuraciones Q×S y conserva la fuente y
el soporte en las identidades auxiliares. Python expande desde la raíz y comparte
algunos nodos estratégicos con el mismo destino, estado CGS, átomo y movimiento.
Además, el generador Python produce soportes suficientes, no necesariamente
mínimos para cualquier fórmula booleana positiva.

Estas diferencias impiden afirmar igualdad literal de arenas o transferir sus
conteos directamente. La prueba general de equivalencia de estas optimizaciones
con esa presentación exacta del texto no se ha añadido aquí. Las comprobaciones
del ejemplo y las pruebas finitas previas acreditan sus casos, no esa afirmación
universal. Se debe documentar el puente o alinear las representaciones antes de
afirmar una correspondencia literal completa.

La validación no modifica las figuras, resultados experimentales ni el PDF
histórico de `docs/thesis.pdf`. La actualización integral de capítulos 4/5 y el
análisis de complejidad real del programa siguen pendientes.
