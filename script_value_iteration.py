import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os

from value_iteration import ValueIterationAgent, evaluate_agent

# Configuración experimentos
MAP_SIZES = ["4x4", "8x8"]
SUCCESS_RATES = np.linspace(0.1, 0.95, 10)

EVAL_EPISODES = 1000
T_MAX = 100
CONVERGENCE_THRESHOLD = 1e-4


# Modificar el succes rate
def modify_env(env, success_rate, goal_reward):
    slip_rate = (1.0 - success_rate) / 2.0

    for state in env.unwrapped.P:
        for action in env.unwrapped.P[state]:
            transitions = env.unwrapped.P[state][action]
            new_transitions = []

            for i, (prob, next_state, reward, done) in enumerate(transitions):
                
                # estados terminales 
                if len(transitions) == 1:
                    new_prob = 1.0
                
                # acción correcta (centro)
                elif i == 1:
                    new_prob = success_rate
                
                # deslizamientos
                else:
                    new_prob = slip_rate
                
                # recompensa personalizada
                new_reward = goal_reward if reward > 0 else reward

                new_transitions.append((new_prob, next_state, new_reward, done))

            env.unwrapped.P[state][action] = new_transitions


# Experimentos
def run_experiments():
    if not os.path.exists("resultados"):
        os.makedirs("resultados")

    try:
        gamma_input = input("Introduce los valores de Gamma, sepáralos por comas [por defecto: 0.95]: ")
        gammas = [float(g.strip()) for g in gamma_input.split(",")] if gamma_input.strip() else [0.95]
        
        reward_input = input("Introduce los valores de señal de recompensa (Goal), sepáralos por comas [por defecto del 1 al 100 en 10 pasos]: ")
        if reward_input.strip():
            rewards_goal = [float(r.strip()) for r in reward_input.split(",")]
        else:
            rewards_goal = np.linspace(1, 100, 10) # Rango de 1 a 100 (10 valores por defecto)
            
    except ValueError:
        gammas = [0.95]
        rewards_goal = np.linspace(1, 100, 10)

    resultados = []

    total_exp = len(MAP_SIZES) * len(SUCCESS_RATES) * len(gammas) * len(rewards_goal)
    print(f"Ejecutando {total_exp} experimentos..\n")

    for r_goal in rewards_goal:
        for gamma in gammas:
            for map_name in MAP_SIZES:
                for sr in SUCCESS_RATES:
                    print(f"Mapa={map_name} | SR={sr:.2f} | Gamma={gamma} | Reward={r_goal:.1f}")
                    
                    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=True)
                    modify_env(env, sr, r_goal)
                    
                    agent = ValueIterationAgent(env, gamma=gamma)
                    
                    # Entrenamiento
                    inicio_train = time.time()
                    iteration = 0
                    while True:
                        _, max_diff = agent.value_iteration()
                        iteration += 1
                        if max_diff < CONVERGENCE_THRESHOLD:
                            break
                    tiempo_train = time.time() - inicio_train

                    # Evaluación
                    eval_rewards = evaluate_agent(agent, env, EVAL_EPISODES, T_MAX)
                    tasa_exito = np.mean([r > 0 for r in eval_rewards]) * 100
                    
                    resultados.append({
                        "Mapa": map_name, 
                        "Success_Rate": sr, 
                        "Gamma": gamma, 
                        "Reward_Goal": r_goal,
                        "Iteraciones": iteration, 
                        "Tiempo_Entrenamiento": tiempo_train, 
                        "Tasa_Exito": tasa_exito
                    })
                    env.close()

    # Exportar los resultados en un csv
    df = pd.DataFrame(resultados)
    df.to_csv("resultados/resultados_completos.csv", index=False)

    # Gráficas
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Success_Rate", y="Tasa_Exito", hue="Mapa", marker="o")
    plt.title("Impacto del Success Rate en la Tasa de Éxito")
    plt.grid()
    plt.savefig("resultados/grafica_exito.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    # Aquí cambié el X para mostrar el Reward_Goal, y usamos Hue para separar por Mapa y Success Rate
    sns.lineplot(data=df, x="Reward_Goal", y="Tasa_Exito", hue="Mapa", marker="o")
    plt.title("Evolución de la Tasa de Éxito vs Señal de Recompensa (1-100)")
    plt.xlabel("Señal de Recompensa en la Meta")
    plt.ylabel("Tasa de éxito (%)")
    plt.grid()
    plt.savefig("resultados/grafica_exito_vs_recompensa.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Success_Rate", y="Iteraciones", hue="Mapa", marker="o")
    plt.title("Convergencia vs Success Rate")
    plt.grid()
    plt.savefig("resultados/grafica_convergencia.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Mapa", y="Tiempo_Entrenamiento", hue="Mapa", errorbar="sd", legend=False)
    plt.title("Comparativa de Escalabilidad: Tiempo Promedio por Mapa")
    plt.grid(axis='y')
    plt.savefig("resultados/grafica_escalabilidad_tiempo.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Gamma", y="Tasa_Exito", hue="Mapa", marker="o")
    plt.title("Impacto de Gamma en la Tasa de Éxito")
    plt.grid()
    plt.savefig("resultados/grafica_gamma.png")
    plt.close()

    print("Las gráficas y el csv están en 'resultados/'")

if __name__ == "__main__":
    run_experiments()