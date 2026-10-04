"""
Clasificación de 15 operadores nuevos con Naive Bayes Gaussiano
(equivalente a fitcnb de MATLAB).

Archivos en la misma carpeta que este script:
  DatosEntrenamiento.xlsx
  DatosNuevos.xlsx

Instalación:
  pip install pandas openpyxl scikit-learn matplotlib numpy

Ejecución:
  python clasificar_operadores.py

Genera en resultados_clasificacion/:
  reporte_clasificacion.pdf        (resumen + 15 hojas con gráficas 2D y 3D)
  resumen_clasificaciones.xlsx
  graficas/operador_XX_2D.png
  graficas/operador_XX_3D.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no abre ventanas; solo guarda archivos
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.naive_bayes import GaussianNB

# ===================== CONFIGURACIÓN =====================
# Aquí se indican los archivos de entrada y la carpeta donde se guardarán los resultados del proyecto
CARPETA = Path(__file__).resolve().parent
ARCHIVO_ENTRENAMIENTO = CARPETA / "DatosEntrenamiento.xlsx"
ARCHIVO_NUEVOS = CARPETA / "DatosNuevos.xlsx"
CARPETA_SALIDA = CARPETA / "resultados_clasificacion"

# Prioridad:
#   A: TiempoCiclo + Defectos
#   B: Defectos + HorasExperiencia      (si falta TiempoCiclo)
#   C: TiempoCiclo + HorasExperiencia   (si faltan Defectos)
# Variabilidad no se usa: la rúbrica no indica relación con las demás.
COMBINACIONES = {
    "A": ("TiempoCiclo", "Defectos"),
    "B": ("Defectos", "HorasExperiencia"),
    "C": ("TiempoCiclo", "HorasExperiencia"),
}
NOMBRES = {
    "TiempoCiclo": "Tiempo de ciclo",
    "Defectos": "Número de defectos",
    "HorasExperiencia": "Horas de experiencia",
}
# Colores de la rúbrica (Fig. 1): Novato rojo, Estándar verde, Experto azul
COLORES = {"Novato": "red", "Estandar": "green", "Experto": "blue"}

# ===================== SELECCIÓN DE VARIABLES =====================
# Esta función revisa qué datos tiene disponibles cada operador y
# selecciona una pareja de variables que pueda utilizarse
def elegir_variables(fila):
    """Devuelve (clave, par de variables) según las variables disponibles."""
    for clave in ("A", "B", "C"):
        par = COMBINACIONES[clave]
        if all(pd.notna(fila[v]) for v in par):
            return clave, par
    raise ValueError("El operador no tiene dos variables disponibles.")

# ===================== CARGA Y VALIDACIÓN DE DATOS =====================
# Se leen los dos archivos Excel y se comprueba que tengan las columnas
# necesarias antes de comenzar la clasificación
def cargar_datos():
    for ruta in (ARCHIVO_ENTRENAMIENTO, ARCHIVO_NUEVOS):
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró {ruta}")

    entrenamiento = pd.read_excel(ARCHIVO_ENTRENAMIENTO)
    nuevos = pd.read_excel(ARCHIVO_NUEVOS)

    base = ["TiempoCiclo", "Defectos", "HorasExperiencia"]
    for col in base + ["Nivel"]:
        if col not in entrenamiento.columns:
            raise ValueError(f"Falta la columna '{col}' en entrenamiento.")
    for col in base:
        if col not in nuevos.columns:
            raise ValueError(f"Falta la columna '{col}' en datos nuevos.")

    for col in base:
        entrenamiento[col] = pd.to_numeric(entrenamiento[col], errors="coerce")
        nuevos[col] = pd.to_numeric(nuevos[col], errors="coerce")
    entrenamiento["Nivel"] = entrenamiento["Nivel"].astype(str).str.strip()

    if entrenamiento[base + ["Nivel"]].isna().any().any():
        raise ValueError("Hay valores faltantes en el entrenamiento.")
    return entrenamiento, nuevos

# ===================== ENTRENAMIENTO =====================
# Aquí se crea y entrena el clasificador Naive Bayes Gaussiano
def entrenar_modelo(entrenamiento, variables):
    """Naive Bayes Gaussiano sin escalar (igual que fitcnb de la plantilla),
    entrenado con las MISMAS variables que se usarán para predecir."""
    modelo = GaussianNB()
    modelo.fit(entrenamiento.loc[:, list(variables)].astype(float),
               entrenamiento["Nivel"])
    return modelo

# ===================== MALLA PARA LAS GRÁFICAS =====================
# Se crea una cuadrícula de puntos para poder visualizar las regiones
# de decisión y las probabilidades del modelo
def preparar_malla(entrenamiento, variables, puntos=140):
    x = entrenamiento[variables[0]].astype(float)
    y = entrenamiento[variables[1]].astype(float)
    dx = max(x.max() - x.min(), 1.0) * 0.08
    dy = max(y.max() - y.min(), 1.0) * 0.08
    xx, yy = np.meshgrid(np.linspace(x.min() - dx, x.max() + dx, puntos),
                         np.linspace(y.min() - dy, y.max() + dy, puntos))
    grid = pd.DataFrame({variables[0]: xx.ravel(), variables[1]: yy.ravel()})
    return xx, yy, grid

# ===================== GRÁFICA 2D =====================
# Esta función muestra los datos de entrenamiento, las zonas de decisión
# y la posición del operador nuevo
def graficar_2d(ax, entrenamiento, idx, fila, variables, modelo,
                xx, yy, grid, prediccion):
    clases = list(modelo.classes_)
    codigos = {c: i for i, c in enumerate(clases)}
    z = np.array([codigos[c] for c in modelo.predict(grid)]).reshape(xx.shape)

    # Mismo color para la región y para los puntos de cada clase
    cmap = ListedColormap([COLORES[c] for c in clases])
    ax.contourf(xx, yy, z, levels=np.arange(-0.5, len(clases) + 0.5, 1),
                alpha=0.22, cmap=cmap)

    for clase in clases:
        d = entrenamiento[entrenamiento["Nivel"] == clase]
        ax.scatter(d[variables[0]], d[variables[1]], s=22, alpha=0.7,
                   color=COLORES[clase], edgecolors="none",
                   label=f"Entrenamiento: {clase}")

    ax.scatter(float(fila[variables[0]]), float(fila[variables[1]]),
               marker="X", s=220, c="black", edgecolors="white",
               linewidths=1.2, zorder=10,
               label=f"Operador {idx + 1}: {prediccion}")
    ax.set_xlabel(NOMBRES[variables[0]])
    ax.set_ylabel(NOMBRES[variables[1]])
    ax.set_title(f"Operador {idx + 1} — Zonas de clasificación 2D")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best")

# ===================== GRÁFICA 3D =====================
# Esta función representa la probabilidad de cada clase como una superficie
def graficar_3d(ax, idx, fila, variables, modelo, xx, yy, grid,
                prediccion, probabilidad):
    posteriores = modelo.predict_proba(grid)
    clases = list(modelo.classes_)
    for j, clase in enumerate(clases):
        ax.plot_surface(xx, yy, posteriores[:, j].reshape(xx.shape),
                        color=COLORES[clase], alpha=0.34, linewidth=0,
                        antialiased=True)
    ax.scatter([float(fila[variables[0]])], [float(fila[variables[1]])],
               [probabilidad], marker="X", s=95, color="black",
               edgecolors="white", linewidths=0.8)
    ax.set_xlabel(NOMBRES[variables[0]], labelpad=8)
    ax.set_ylabel(NOMBRES[variables[1]], labelpad=8)
    ax.set_zlabel("Probabilidad", labelpad=10)
    ax.set_zlim(0, 1)
    ax.set_title(f"Operador {idx + 1} — Probabilidad de clasificación 3D\n"
                 f"{prediccion} ({probabilidad:.4f})")
    handles = [Patch(facecolor=COLORES[c], alpha=0.5, label=c) for c in clases]
    ax.legend(handles=handles, fontsize=8, loc="upper left")

# ===================== RESUMEN DEL PDF =====================
# Crea la primera página del reporte con la tabla de resultados
def pagina_resumen(pdf, resumen):
    """Tabla con operador, variables usadas, clasificación y probabilidad."""
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.axis("off")
    ax.set_title("Clasificación de los 15 operadores nuevos — Naive Bayes "
                 "Gaussiano", fontsize=14, fontweight="bold")
    filas = [[int(r["Operador"]),
              f'{r["Variable 1"]} + {r["Variable 2"]}',
              f'{r["Valor variable 1"]:g} , {r["Valor variable 2"]:g}',
              r["Clasificación predicha"],
              f'{r["Probabilidad de la clasificación"]:.4f}']
             for _, r in resumen.iterrows()]
    tabla = ax.table(
        cellText=filas,
        colLabels=["Operador", "Variables usadas", "Valores ingresados",
                   "Clasificación", "Probabilidad"],
        loc="center", cellLoc="center",
        colWidths=[0.09, 0.38, 0.17, 0.16, 0.14])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.6)
    for (fila, _), celda in tabla.get_celld().items():
        if fila == 0:
            celda.set_facecolor("#d9e2f3")
            celda.set_text_props(fontweight="bold")
    pdf.savefig(fig)
    plt.close(fig)

# ===================== PROCESO PRINCIPAL =====================
# main(): carga datos, entrena, predice, grafica y guarda los resultados
def main():

    # 1) Carga y valida los datos de entrenamiento y los operadores nuevos
    entrenamiento, nuevos = cargar_datos()
    if len(nuevos) != 15:
        print(f"AVISO: se esperaban 15 operadores; hay {len(nuevos)}.")

    # Crea la carpeta donde se almacenarán las gráficas
    carpeta_graficas = CARPETA_SALIDA / "graficas"
    carpeta_graficas.mkdir(parents=True, exist_ok=True)

    # listas/diccionarios almacenarán los resultados durante el proceso
    resultados, modelos, figuras_pdf = [], {}, []

    # 2) Procesa cada operador nuevo individualmente
    for idx, fila in nuevos.iterrows():
        # Elige las dos variables disponibles para el operador
        clave, variables = elegir_variables(fila)

        # Si todavía no existe un modelo para esta combinación de variables,
        # lo entrenamos 
        if clave not in modelos:
            modelos[clave] = entrenar_modelo(entrenamiento, variables)
        # Recuperamos el modelo correspondiente
        modelo = modelos[clave]

        # Creamos el registro del operador nuevo usando exactamente
        # las mismas variables con las que se entrenó el modelo
        X_nuevo = pd.DataFrame([[float(fila[v]) for v in variables]],
                               columns=list(variables))
        
        # 3) predict() devuelve la categoría predicha
        etiqueta = str(modelo.predict(X_nuevo)[0])

        # 4) predict_proba() devuelve las probabilidades de todas las clases
        posterior = modelo.predict_proba(X_nuevo)[0]

        # Asociamos cada probabilidad con su nombre de clase
        probas = dict(zip(modelo.classes_, posterior))

        # Obtenemos la probabilidad correspondiente a la clase predicha
        prob = float(probas[etiqueta])

        # 5) Guardamos los resultados del operador:
        resultados.append({
            "Operador": idx + 1,
            "Combinación": clave,
            "Variable 1": NOMBRES[variables[0]],
            "Valor variable 1": float(fila[variables[0]]),
            "Variable 2": NOMBRES[variables[1]],
            "Valor variable 2": float(fila[variables[1]]),
            "Clasificación predicha": etiqueta,
            "Probabilidad de la clasificación": prob,
            "Probabilidad Novato": float(probas.get("Novato", np.nan)),
            "Probabilidad Estandar": float(probas.get("Estandar", np.nan)),
            "Probabilidad Experto": float(probas.get("Experto", np.nan)),
        })

        # 6) Crea la malla para las gráficas
        xx, yy, grid = preparar_malla(entrenamiento, variables)
        nombre = f"operador_{idx + 1:02d}"

        # 7) Genera y guarda la gráfica 2D individual
        f2 = plt.figure(figsize=(8, 6))
        graficar_2d(f2.add_subplot(111), entrenamiento, idx, fila, variables,
                    modelo, xx, yy, grid, etiqueta)
        f2.tight_layout()
        f2.savefig(carpeta_graficas / f"{nombre}_2D.png", dpi=160,
                   bbox_inches="tight")
        plt.close(f2)

        # 8) Genera y guarda la gráfica 3D individual
        f3 = plt.figure(figsize=(8, 6.5))
        graficar_3d(f3.add_subplot(111, projection="3d"), idx, fila,
                    variables, modelo, xx, yy, grid, etiqueta, prob)
        f3.savefig(carpeta_graficas / f"{nombre}_3D.png", dpi=160,
                   bbox_inches="tight", pad_inches=0.4)
        plt.close(f3)

        # 9) Crea la página combinada (2D + 3D) que se incluirá en el PDF
        fig = plt.figure(figsize=(14, 6.4))
        graficar_2d(fig.add_subplot(1, 2, 1), entrenamiento, idx, fila,
                    variables, modelo, xx, yy, grid, etiqueta)
        graficar_3d(fig.add_subplot(1, 2, 2, projection="3d"), idx, fila,
                    variables, modelo, xx, yy, grid, etiqueta, prob)
        fig.suptitle(
            f"Operador {idx + 1} | Variables: {NOMBRES[variables[0]]} + "
            f"{NOMBRES[variables[1]]} | Clasificación: {etiqueta} | "
            f"Probabilidad: {prob:.4f}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        figuras_pdf.append(fig)
        # También mostramos el resultado en la terminal
        print(f"Operador {idx + 1:02d}: {etiqueta:8s} | {prob:.4f} | "
              f"{NOMBRES[variables[0]]} + {NOMBRES[variables[1]]}")
        
    # 10) Convertimos los resultados a una tabla y la guardamos en Excel
    resumen = pd.DataFrame(resultados)
    resumen.to_excel(CARPETA_SALIDA / "resumen_clasificaciones.xlsx",
                     index=False)

    # Un solo PDF: primero la tabla resumen, luego las 15 hojas de gráficas
    with PdfPages(CARPETA_SALIDA / "reporte_clasificacion.pdf") as pdf:
        pagina_resumen(pdf, resumen)
        for fig in figuras_pdf:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nListo. Resultados en: {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()
