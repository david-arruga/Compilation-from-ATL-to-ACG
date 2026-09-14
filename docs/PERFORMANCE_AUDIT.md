# Coste de Python y reproducibilidad del capítulo 5

Código analizado: rama corregida, padre `2e2c9c777ce2beda53c9c9c819099d3b42414b01`.
Este bloque añade diagnóstico y datos nuevos; no modifica el compilador.

## Hallazgo principal: la implementación no es lineal en el peor caso

Fijamos un solo agente a, una proposición p y un CGS constante. Sea
φ₀=p y φⱼ₊₁=⟨a⟩Gφⱼ. Su AST tiene 2n+1 nodos y el ACG core 2n+2 estados.

En `generate_closure`, para cada sufórmula estratégica φⱼ se ejecuta
`deepcopy(φⱼ)` antes de normalizar su negación. Cada copia recorre 2j+1 nodos.
Se copia además el literal una vez. Por tanto, solo estas copias recorren

    1 + Σ(j=1..n)(2j+1) = (n+1)²

ocurrencias AST. Las distintas invocaciones de deepcopy no comparten su memo.
Esto da una cota inferior cuadrática de trabajo de copia en esta familia,
bajo un coste unitario por nodo visitado. No es una prueba de cota superior
cuadrática del programa entero: hay también hashing recursivo, igualdad,
normalizaciones y manejo de tablas.

El contador instrumentado confirma exactamente la fórmula anterior. Los tiempos
siguientes son medianas de tres ejecuciones locales, no una prueba asintótica:

| n | Nodos AST | Estados ACG | Nodos copiados | Construcción ACG, ms |
|---:|---:|---:|---:|---:|
| 8 | 17 | 18 | 81 | 0.828 |
| 16 | 33 | 34 | 289 | 3.084 |
| 32 | 65 | 66 | 1089 | 13.388 |
| 64 | 129 | 130 | 4225 | 55.351 |

El teorema de construcción lineal se refiere a una representación simbólica
con referencias e instrumentación apropiadas. No certifica esta construcción
basada en copiar fórmulas como claves de conjuntos y diccionarios. El número
lineal de estados no implica tiempo lineal para producirlos.

## Crecimiento del modelo frente al tamaño de la fórmula

La familia lights con n agentes crea 2ⁿ estados, 2ⁿ decisiones conjuntas por
estado y 4ⁿ entradas de transición. Estos conteos salen directamente de los
bucles de `generate_lights_cgs`. Una curva frente a n no es una curva frente
al tamaño explícito del CGS. Un tiempo exponencial en n puede ser polinómico
en el tamaño del modelo construido.

Muestra diagnóstica de flatG, tres repeticiones por tamaño:

| n | Estados CGS | Transiciones CGS | Vértices juego | Aristas juego | Arena, ms | Solver, ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 4 | 14 | 18 | 0.307 | 1.276 |
| 2 | 4 | 16 | 68 | 101 | 2.008 | 11.722 |
| 3 | 8 | 64 | 294 | 488 | 15.275 | 164.857 |

Todas estas fórmulas positivas G son falsas inicialmente, pues todos los p_i
comienzan falsos. Es un control semántico sencillo, no un benchmark variado
ni suficiente para caracterizar todas las cargas del solver.

## Qué se mide y cómo reproducirlo

```bash
PYTHONHASHSEED=0 python3 -m scripts.performance_audit --output docs/performance_sample.json
```

El archivo JSON contiene 21 observaciones individuales, Python/plataforma y
hashes SHA-256 de los módulos de producción. Se miden por separado generación
del CGS, normalización, construcción ACG, arena y resolución. Los datos incluyen
conteos estructurales y resultados de satisfacción. No hay parseado medido:
los generadores entregan AST. Generar la fórmula, imprimir resultados y contar
copias se hace fuera de los intervalos medidos. La arena incluye la validación
CGS y se construye desde la raíz, conservando los sumideros. El alfabeto es
simbólico. No se mide memoria ni se desactiva el recolector durante cada fase.

Las copias se instrumentan en una ejecución separada; los tiempos no utilizan
el wrapper del contador. Si cambia la implementación y deja de cumplirse la
fórmula de copias, el diagnóstico falla y deberá actualizarse. Los tiempos
variarán por máquina, versión, ruido del entorno y política del recolector.
La muestra no autoriza regresiones asintóticas ni comparaciones con otros tools.

## Estado de los experimentos antiguos

En el repositorio inspeccionado se encuentran generadores, ejemplos, smoke
scripts y el PDF. No se han localizado CSV/JSON de resultados experimentales,
notebooks o scripts que regeneren las figuras del capítulo 5. Esto no significa
que el autor no los conserve fuera del repositorio. Con este material no podemos
reproducir exactamente las figuras ni sus muestras, semillas y mediciones.
El nuevo JSON no sustituye ni reproduce aquellas observaciones.

No se modifica el capítulo 5 ni se declaran falsos sus tiempos históricos.
Sí debe evitarse interpretar la tendencia observada como prueba del peor caso
lineal del Python: el trabajo de copia anterior impide esa conclusión.

## Siguiente cambio recomendado

Para aspirar a la complejidad demostrada hay que dejar de copiar subfórmulas
completas como identidad de cada estado: construir estados por índices y
referencias compartidas, con polaridad explícita, y justificar su decodificación
hacia las fórmulas. La impresión de fórmulas debe quedar fuera del coste de
compilar. Eliminar únicamente deepcopy no basta, pues persisten hashing e
igualdad recursivos y normalizaciones repetidas; cachear hashes de objetos
mutables sin otro contrato tampoco es una solución segura.

El solver también recalcula sucesores recorriendo aristas y usa atractores por
iteración completa. Las cotas de un atractor eficiente con listas de adyacencia
y contadores no se transfieren automáticamente a ese código. Primero interesa
alinear la representación del constructor; después medir y optimizar cada fase
sin modificar la semántica ni confundir las cotas de sus distintas entradas.
