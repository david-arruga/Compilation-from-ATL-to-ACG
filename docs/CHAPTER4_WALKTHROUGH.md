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

## Arena completa y exploración desde la raíz

La implementación actual conserva la configuración de origen y el soporte en
los vértices de átomo y movimiento; usa todos los soportes mínimos. Se han
retirado las dos diferencias de representación detectadas en el cotejo anterior.

```bash
python3 main.py --formula '<Valve> globally (underpowered implies <Reactor> next efficient)' --cgs 1 --full-arena
```

La arena completa tiene **152 vértices y 216 aristas**, sobre los 10 estados
core ACG y 5 estados CGS. Sin la opción se explora desde la raíz y se conservan
ambos sumideros: **77 vértices y 109 aristas**. No deben mezclarse estas medidas.
Ambas construcciones satisfacen la fórmula desde start.

### Justificación de la restricción alcanzable

Sea J la arena completa y R el conjunto de vértices alcanzables desde la raíz,
aumentado con los dos sumideros. R es cerrado por sucesores: un sucesor de un
vértice alcanzable también es alcanzable, y cada sumidero solo se sucede a sí
mismo. Las reglas de expansión dependen del vértice completo y de las entradas,
no del orden de visita. Por inducción sobre la longitud de los caminos, la
exploración visita todos los vértices alcanzables y no introduce otros, salvo
los sumideros declarados. Conserva exactamente aristas, propietarios y Büchi
sobre R. Así, las jugadas desde la raíz son las mismas. Una estrategia completa
se restringe a R; una estrategia en R se extiende arbitrariamente fuera, donde
hay movimientos legales por totalidad. La condición de aceptación no cambia.
Este es un argumento matemático de implementación, no un nuevo teorema Lean.

### Justificación de los soportes mínimos

Antes de minimizar, la enumeración tiene la propiedad: una asignación H satisface
la transición si y solo si contiene algún soporte enumerado. Se prueba por
inducción: Top enumera el vacío, Bottom ninguno, un átomo su singleton; disyunción
concatena alternativas y conjunción une cada par de soportes. Si un soporte es
mínimo entre los enumerados pero tuviera un subconjunto satisfactorio propio,
la propiedad anterior daría un soporte enumerado aún menor, contradicción.
Recíprocamente todo soporte satisfactorio mínimo debe ser enumerado. Eliminar
duplicados y superconjuntos propios produce exactamente los soportes mínimos.
La enumeración y minimización pueden ser costosas; no se afirma coste lineal.

Se comprueban la identidad de las restricciones en tres fórmulas y la separación
de vértices en un caso donde distintos soportes comparten un átomo. Las pruebas
no sustituyen una verificación formal universal del programa.

El PDF histórico de `docs/thesis.pdf` y MAIN_REVISADO no se modifican en este
bloque. Los resultados experimentales siguen pendientes de reproducción.
