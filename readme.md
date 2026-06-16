# Proyecto de Aprendizaje por Refuerzo con Agentes de IA: Frozen Lake Estocástico

Este proyecto implementa y compara diferentes algoritmos de aprendizaje por refuerzo (Reinforcement Learning) para resolver el entorno **FrozenLake-v1** en su modo estocástico (*slippery*). El objtivo principal es analizar cómo afecta la complejidad del entorno, los métodos utilizados y la parametrización al rendimiento del agente en el entorno en el cúal actúa.

<p align="center">
  <img src="frozen_lake.gif" alt="Agente navegando por el lago helado" width="400">
  <br>
  <em>Un vistazo al entorno Frozen Lake</em>
</p>

## Descripción

En esta práctica hemos diseñado y ejecutado una serie de experimentos para estudiar empíricamente el comportamiento de agentes con reinforcment learning. Nos hemos centrado en los siguientes métodos fundamentales:

1. **Iteración de Valor (Value Iteration)**
2. **Estimación Directa (Model-Based)**
3. **Q-Learning (Model-Free)**

Además, hemos explorado cómo escala el rendimiento de estos algoritmos en función de:
* El tamaño del mapa (4x4, 8x8).
* La estocasticidad del entorno (variando la probabilidad de transición o `success_rate`).
* Los parámetros específicos de cada algoritmo ($\gamma$ `Gamma`, $\alpha$ `Alpha`, $\epsilon$ `Epsilon`, etc...).

Para más detalles sobre las decisiones de diseño de los experimentos, hipótesis iniciales y el análisis crítico comparativo, consulta el documento **`documentacion.pdf`** incluido en la entrega.

## Requisitos e Instalación

Este proyecto ha sido desarrollado y probado utilizando **Python 3.11**.

Te recomendamos crear un entorno virtual para aislar las dependencias del proyecto. Puedes crear y activar el entorno (e instalar las dependencias) ejecutando los siguientes comandos:

1. Crear entorno virtual con Python 3.11 usando conda:
  ```bash
  conda create -n mi_entorno python=3.11
  ```
2. Activa el entorno:
  ```bash
  conda activate mi_entorno
  ```
3. Instala las dependencias:
  ```bash
  pip install -r requeriments.txt
  ```

4. Ejecuta en la terminal el siguiente comando para probar los experimentos documentados:
  ```bash
  python script.py
  ```

> [!TIP]
> En los algoritmos, si se usan muchos parámetros diferentes, la ejecución puede tardar mucho.


