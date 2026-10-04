# Clasificación de operadores con Naive Bayes Gaussiano

## 1. Descripción del proyecto

Este proyecto realiza la **clasificación de 15 operadores nuevos**
utilizando un modelo de **Naive Bayes Gaussiano** en Python.

El programa toma como referencia `DatosEntrenamiento.xlsx` y utiliza
`DatosNuevos.xlsx` para clasificar a cada operador en una de las
siguientes categorías:

-   **Novato**
-   **Estandar**
-   **Experto**

La implementación sigue la lógica de la plantilla de MATLAB, donde se
utiliza `fitcnb`. En Python se utiliza `GaussianNB`.

## 2. Objetivo

El programa permite obtener para cada uno de los 15 operadores:

1.  Las variables utilizadas para la clasificación.
2.  La clasificación predicha.
3.  La probabilidad asociada a la clasificación.
4.  Una gráfica 2D de las zonas de clasificación.
5.  Una gráfica 3D de las probabilidades de clasificación.

## 3. Datos utilizados

### DatosEntrenamiento.xlsx

Contiene los datos utilizados para entrenar el modelo:

-   `TiempoCiclo`: tiempo de ciclo.
-   `Defectos`: número de defectos.
-   `HorasExperiencia`: horas de experiencia.
-   `Nivel`: categoría del operador.

### DatosNuevos.xlsx

Contiene los datos de los 15 operadores que se desean clasificar. Se
utilizan las variables disponibles entre:

-   `TiempoCiclo`
-   `Defectos`
-   `HorasExperiencia`

La variable `Variabilidad` no se utiliza porque el planteamiento
empleado no establece una relación de esta variable con las demás.

## 4. Variables utilizadas

El programa contempla tres combinaciones:

### Combinación A

**Tiempo de ciclo + Número de defectos**

Es la combinación principal y corresponde a la lógica de la plantilla
original de MATLAB.

### Combinación B

**Número de defectos + Horas de experiencia**

Se utiliza cuando no está disponible el tiempo de ciclo.

### Combinación C

**Tiempo de ciclo + Horas de experiencia**

Se utiliza cuando no está disponible el número de defectos.

El programa selecciona la primera combinación cuyos dos valores estén
disponibles para el operador.

## 5. Modelo de clasificación

Se utiliza **Naive Bayes Gaussiano** mediante:

``` python
GaussianNB()
```

El modelo se entrena con las variables seleccionadas y con `Nivel` como
variable objetivo:

``` python
modelo = GaussianNB()
modelo.fit(
    entrenamiento.loc[:, list(variables)].astype(float),
    entrenamiento["Nivel"]
)
```

Esto corresponde conceptualmente al uso de:

``` matlab
fitcnb(X,Y)
```

en MATLAB.

## 6. Predicción

Para cada operador se construye un registro con las mismas dos variables
utilizadas durante el entrenamiento.

La clase se obtiene mediante:

``` python
modelo.predict(X_nuevo)
```

El resultado puede ser `Novato`, `Estandar` o `Experto`.

Es importante utilizar durante la predicción las mismas variables con
las que se entrenó el modelo.

## 7. Probabilidades

Las probabilidades se obtienen mediante:

``` python
modelo.predict_proba(X_nuevo)
```

El modelo calcula una probabilidad para cada categoría:

-   `Novato`
-   `Estandar`
-   `Experto`

Después se toma la probabilidad correspondiente a la clase predicha.

## 8. Gráficas 2D

Para cada operador se genera una gráfica 2D que muestra:

-   Los datos del conjunto de entrenamiento.
-   Las tres categorías.
-   Las zonas de decisión del modelo.
-   El operador nuevo marcado con una `X`.

Para construir las zonas se genera una malla mediante:

``` python
np.meshgrid(...)
```

y se predice la clase en cada punto de la malla.

Los colores utilizados son:

-   Rojo: `Novato`
-   Verde: `Estandar`
-   Azul: `Experto`

Estas gráficas permiten visualizar las regiones donde el modelo
considera que pertenece cada categoría.

## 9. Gráficas 3D

Para cada operador también se genera una gráfica 3D de probabilidad.

Se utiliza:

``` python
modelo.predict_proba(grid)
```

para obtener las probabilidades en los puntos de una malla.

Después se representan las superficies mediante:

``` python
ax.plot_surface(...)
```

El eje Z representa la **probabilidad de clasificación**, con valores
entre 0 y 1.

El operador nuevo se muestra como una `X` sobre la probabilidad
correspondiente a su clasificación.

## 10. Proceso general

El funcionamiento del programa puede resumirse así:

1.  Localiza los archivos de entrenamiento y datos nuevos.
2.  Carga los datos de Excel.
3.  Comprueba que existan las columnas necesarias.
4.  Convierte las variables numéricas al formato adecuado.
5.  Determina qué combinación de variables puede utilizar cada operador.
6.  Entrena el modelo Naive Bayes.
7.  Obtiene la clasificación mediante `predict()`.
8.  Obtiene las probabilidades mediante `predict_proba()`.
9.  Genera una gráfica 2D.
10. Genera una gráfica 3D.
11. Guarda los resultados en Excel.
12. Genera un reporte PDF.

## 11. Archivos generados

Al ejecutar el programa se crea:

``` text
resultados_clasificacion/
```

Dentro se generan:

### `reporte_clasificacion.pdf`

Incluye una tabla resumen y una página para cada operador con sus
gráficas 2D y 3D.

### `resumen_clasificaciones.xlsx`

Incluye:

-   Operador.
-   Combinación utilizada.
-   Variables utilizadas.
-   Valores de las variables.
-   Clasificación predicha.
-   Probabilidad de la clasificación.
-   Probabilidad de `Novato`.
-   Probabilidad de `Estandar`.
-   Probabilidad de `Experto`.

### Carpeta `graficas/`

Contiene las 15 gráficas 2D y las 15 gráficas 3D:

``` text
operador_01_2D.png
operador_01_3D.png
...
operador_15_2D.png
operador_15_3D.png
```

## 12. Requisitos

Se requiere Python 3 y las siguientes bibliotecas:

``` bash
pip install pandas openpyxl scikit-learn matplotlib numpy
```

Funciones principales:

-   `pandas`: lectura y manejo de Excel.
-   `openpyxl`: soporte para `.xlsx`.
-   `scikit-learn`: modelo Naive Bayes.
-   `matplotlib`: gráficas y PDF.
-   `numpy`: generación de mallas.

## 13. Ejecución

Los tres archivos principales deben estar en la misma carpeta:

``` text
clasificar_operadores.py
DatosEntrenamiento.xlsx
DatosNuevos.xlsx
```

Ejecutar desde una terminal:

``` bash
python clasificar_operadores.py
```

Al finalizar se generará la carpeta `resultados_clasificacion/`.

## 14. Relación con MATLAB

La implementación mantiene la lógica principal de la plantilla:

  MATLAB                   Python
  ------------------------ ---------------------------------
  `fitcnb(X,Y)`            `GaussianNB().fit(X,Y)`
  `predict(mdl,X_nuevo)`   `modelo.predict(X_nuevo)`
  `Posterior`              `modelo.predict_proba(X_nuevo)`
  `meshgrid`               `np.meshgrid`
  `surf`                   `ax.plot_surface`

Por lo tanto, Python reproduce las etapas principales de entrenamiento,
predicción, cálculo de probabilidades y visualización de la plantilla de
MATLAB.

## 15. Consideración sobre las probabilidades

Las probabilidades reportadas son **probabilidades posteriores
calculadas por el modelo Naive Bayes**.

## 16. Conclusión

El programa permite clasificar los 15 operadores nuevos mediante Naive
Bayes Gaussiano y presentar los resultados de manera numérica y gráfica.

Para cada operador se obtiene una clasificación y su probabilidad
correspondiente, además de una representación 2D de las zonas de
decisión y una representación 3D de las probabilidades.

