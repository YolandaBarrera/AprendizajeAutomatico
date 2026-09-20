# Tarea 3 - Límites de Control
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path 
 
# Busca el Excel 
archivo = Path(__file__).parent / "Dataset_Ejercicio_Limites_de_Control.xlsx" 
 
# Pandas abre el Excel y almacena sus datos en un df 
df = pd.read_excel(archivo) 
 
# Toma la columna: Monthly Wage y la guarda en una variable llamada:salarios 
salarios = df["Monthly Wage"] 
 
# ================================================================ 
# 2. Ordenar los salarios y dividirlos en 4 grupos de igual tamaño 
# ================================================================ 
 
# ordena todos los registros de menor a mayor, Y reinicia los índices 
ordenado = df.sort_values("Monthly Wage").reset_index(drop=True) 
 
n = len(ordenado) # cuenta cuántos registros hay 
tam_cuartil = n // 4 # Tamaño de cada cuartil 
 
print("\n========== CUARTILES ==========") 
# Imprime los cuartiles 
for i in range(4): 
    inicio = i * tam_cuartil # Inicio del cuartil 
    fin = (i + 1) * tam_cuartil - 1 # Fin del cuartil 
 
    print(f"\nCuartil {i + 1}") 
    print( 
        f"Inicio: elemento {inicio + 1} | " 
        f"{ordenado.loc[inicio, 'State']} | " 
        f"${ordenado.loc[inicio, 'Monthly Wage']:,.2f}" 
    ) 
    print( 
        f"Final:  elemento {fin + 1} | " 
        f"{ordenado.loc[fin, 'State']} | " 
        f"${ordenado.loc[fin, 'Monthly Wage']:,.2f}" 
    ) 
 
# Límites de la ‘caja’ conformada por el último valor del cuartil 1 
#  y conformada por el último valor del cuartil 3. 
limite_caja_inferior = ordenado.loc[tam_cuartil - 1, "Monthly Wage"] 
limite_caja_superior = ordenado.loc[3 * tam_cuartil - 1, "Monthly Wage"] 
 
# ============================================================ 
# 3. Media, varianza y desviación estándar 
# ============================================================ 
 
media = salarios.mean() 
 
# Se utiliza varianza y desviación estándar poblacionales (ddof=0) 
# porque se están analizando los 32 estados incluidos en el dataset. 
varianza = salarios.var(ddof=0) 
desviacion = salarios.std(ddof=0) 
 
print("\n========== ESTADÍSTICA DESCRIPTIVA ==========") 
print(f"Media:                ${media:,.2f}") 
print(f"Varianza poblacional: {varianza:,.2f}") 
print(f"Desviación estándar:  ${desviacion:,.2f}") 
 
# ============================================================ 
# Estado cuyo salario está más cerca de la media 
# ============================================================ 
 
# Calcula la distancia absoluta de cada salario respecto a la media 
distancias = abs(salarios - media) 
# busca el índice correspondiente a la distancia más pequeña 
indice = distancias.idxmin() 
 
print("\n========== SALARIO MÁS CERCANO A LA MEDIA ==========") 
print(f"Estado:  {df.loc[indice, 'State']}") 
print(f"Salario: ${df.loc[indice, 'Monthly Wage']:,.2f}") 
print(f"Distancia respecto a la media: ${distancias.loc[indice]:,.2f}") 
 
# ============================================================ 
# 4. Límites de control 
# ============================================================ 
 
limites = {} 
 
print("\n========== LÍMITES DE CONTROL ==========") 
 
for k in [1, 2, 3]: 
    limite_inferior = media - k * desviacion 
    limite_superior = media + k * desviacion 
 
    limites[k] = (limite_inferior, limite_superior) 
 
    print(f"\n{k} desviación(es) estándar:") 
    print(f"Límite inferior: ${limite_inferior:,.2f}") 
    print(f"Límite superior: ${limite_superior:,.2f}") 
 
# ============================================================ 
# 5. Histograma 
# ============================================================ 
 
plt.figure(figsize=(8, 4)) 
 
plt.hist(salarios, bins=8, edgecolor="black", color="skyblue") 
# Línea de la media 
plt.axvline( 
    media, 
    color="red",
    linestyle="--", 
    linewidth=2, 
    label=f"Media = ${media:,.2f}" 
) 
# Línea de Q1  
plt.axvline( 
    limite_caja_inferior, 
    color="green",
    linestyle=":", 
    linewidth=2, 
    label=f"Final Q1 = ${limite_caja_inferior:,.2f}" 
) 
# Línea de Q3 
plt.axvline( 
    limite_caja_superior, 
    color="purple",
    linestyle=":", 
    linewidth=2, 
    label=f"Final Q3 = ${limite_caja_superior:,.2f}" 
) 
 
plt.title("Histograma de salarios mensuales") 
plt.xlabel("Salario mensual") 
plt.ylabel("Frecuencia") 
plt.legend() 
plt.grid(axis="y", alpha=0.3) 
plt.tight_layout() 
# Guarda el histograma 
plt.savefig("histograma_salarios.png", dpi=300) 
plt.show(block=False)
 
# ============================================================ 
# 6. Boxplot / gráfica de control 
# ============================================================ 
 
limite_inferior_3sigma, limite_superior_3sigma = limites[3] 
 
# Se construye el boxplot manualmente para que Q1 y Q3 sean 
# exactamente los últimos valores de los grupos 1 y 3. 
stats = [{ 
    "label": "Salarios", 
    "whislo": salarios.min(), 
    "q1": limite_caja_inferior, 
    "med": salarios.median(), 
    "q3": limite_caja_superior, 
    "whishi": salarios.max(), 
    "fliers": [] 
}] 
# Crea el Boxplot 
fig, ax = plt.subplots(figsize=(8, 6)) 
 
ax.bxp(
    stats, 
    showfliers=False, 
    vert=True,
    boxprops=dict(color="steelblue", linewidth=2),
    whiskerprops=dict(color="steelblue", linewidth=2),
    capprops=dict(color="steelblue", linewidth=2),
    medianprops=dict(color="orange", linewidth=2)
) 
 
# Línea de la media 
ax.axhline( 
    media, 
    color="red",
    linestyle="--", 
    linewidth=2, 
    label=f"Media = ${media:,.2f}" 
) 
 
# Límites de control de +/- 3 desviaciones estándar 
ax.axhline( 
    limite_inferior_3sigma, 
    color="blue",
    linestyle=":", 
    linewidth=2, 
    label=f"Límite inferior (-3σ) = ${limite_inferior_3sigma:,.2f}" 
) 
 
ax.axhline( 
    limite_superior_3sigma, 
    color="blue",
    linestyle=":", 
    linewidth=2, 
    label=f"Límite superior (+3σ) = ${limite_superior_3sigma:,.2f}" 
) 
 
# Límites de la caja 
ax.axhline( 
    limite_caja_inferior, 
    color="green",
    linestyle="-.", 
    linewidth=1.5, 
    label=f"Final Q1 = ${limite_caja_inferior:,.2f}" 
) 
 
ax.axhline( 
    limite_caja_superior, 
    color="purple",
    linestyle="-.", 
    linewidth=1.5, 
    label=f"Final Q3 = ${limite_caja_superior:,.2f}" 
) 
 
ax.set_title("Gráfica de control / Boxplot de salarios") 
ax.set_ylabel("Salario mensual") 
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1)) 
ax.grid(axis="y", alpha=0.3) 
 
plt.tight_layout() 
# Guarda el Boxplot 
plt.savefig("boxplot_limites_control.png", dpi=300) 
plt.show(block=False)
 

# ============================================================
# 6.1 Gráfica de dispersión Workforce vs salario
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Workforce"],
    df["Monthly Wage"],
    color="orange",
    edgecolor="black",
    s=70
)

plt.title("Relación entre fuerza laboral y salario mensual")
plt.xlabel("Fuerza laboral")
plt.ylabel("Salario mensual")
plt.grid(alpha=0.3)
plt.tight_layout()

# Guarda la gráfica de dispersión
plt.savefig("workforce_vs_salario.png", dpi=300)
plt.show(block=False)

# ============================================================
# 6.2 Salario promedio por estado
# ============================================================

# Calcula el salario promedio de cada estado
salario_por_estado = df.groupby("State")["Monthly Wage"].mean().sort_values()

plt.figure(figsize=(8, 6))

plt.barh(
    salario_por_estado.index,
    salario_por_estado.values,
    color="steelblue",
    edgecolor="black"
)

plt.title("Salario mensual promedio por estado")
plt.xlabel("Salario mensual promedio")
plt.ylabel("Estado")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

# Guarda la gráfica
plt.savefig("salario_promedio_por_estado.png", dpi=300)
plt.show()

# ============================================================ 
# 7. Mostrar todos los datos ordenados 
# ============================================================ 
 
# Tabla final de datos ordenados 
tabla = ordenado[["State", "Monthly Wage"]].copy() 
# agrega una columna llamada: Elemento y numera los registros 
tabla.insert(0, "Elemento", range(1, len(tabla) + 1)) 
 
print("\n========== DATOS ORDENADOS ==========") 
print(tabla.to_string(index=False)) 
 
print("\nSe generaron:") 
print("1. histograma_salarios.png") 
print("2. boxplot_limites_control.png")
print("3. workforce_vs_salario.png")
print("4. salario_promedio_por_estado.png")

