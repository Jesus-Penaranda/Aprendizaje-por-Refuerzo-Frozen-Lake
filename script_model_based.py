import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
import copy

from model_based import DirectEstimationAgent

# Configuración experimentos
MAP_SIZES = ["4x4", "8x8"]
SUCCESS_RATES = np.linspace(0.1, 0.95, 10)

EVAL_EPISODES = 10000
T_MAX = 1000
CONVERGENCE_THRESHOLD = 1e-4
MAX_ITERATIONS = 10000
NUM_TRAJECTORIES = 100

# Modificar el succes rate
def modify_slip_rate(env, success_rate, goal_reward):
    slip_rate = (1.0 - success_rate) / 2.0
    original_P = copy.deepcopy(env.unwrapped.P)
    for state in original_P:
        for action in original_P[state]:
            transitions = original_P[state][action]
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


def evaluate_agent(agent, env, episodes, t_max):
    rewards = []
    for _ in range(episodes):
        total_reward = 0.0
        state, _ = env.reset()
        for _ in range(t_max):
            action = agent.select_action(state)
            new_state, new_reward, is_done, truncated, _ = env.step(action)
            total_reward += new_reward
            if is_done or truncated:
                break
            state = new_state
        rewards.append(total_reward)
    return rewards


# Experimentos
def run_experiments():
    output_dir = "resultados_mb"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        gamma_input = input("Introduce los valores de Gamma para Model-Based, separálos por comas [por defecto: 0.95]: ")
        if gamma_input.strip() == "":
            gammas = [0.95]
        else:
            gammas = [float(g.strip()) for g in gamma_input.split(",")]
            
        reward_input = input("Introduce los valores de señal de recompensa (Goal), sepáralos por comas [por defecto del 1 al 100 en 10 pasos]: ")
        if reward_input.strip():
            rewards_goal = [float(r.strip()) for r in reward_input.split(",")]
        else:
            rewards_goal = np.linspace(1, 100, 10)
    except ValueError:
        gammas = [0.95]
        rewards_goal = np.linspace(1, 100, 10)

    resultados = []
    total_exp = len(MAP_SIZES)*len(SUCCESS_RATES)*len(gammas)*len(rewards_goal)
    print(f"Ejecutando {total_exp} experimentos para Model-Based..\n")

    for r_goal in rewards_goal:
        for gamma in gammas:
            for map_name in MAP_SIZES:
                for sr in SUCCESS_RATES:
                    print(f"Mapa={map_name} | SR={sr:.2f} | Gamma={gamma} | Reward={r_goal:.1f}")
                    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=True)
                    modify_slip_rate(env, sr, r_goal)
                    agent = DirectEstimationAgent(env, gamma=gamma, num_trajectories=NUM_TRAJECTORIES)
                    
                    # Entrenamiento
                    inicio_train = time.time()
                    iteration = 0
                    while iteration < MAX_ITERATIONS:
                        _, max_diff = agent.value_iteration()
                        iteration += 1
                        if max_diff < CONVERGENCE_THRESHOLD:
                            break
                    tiempo_train = time.time() - inicio_train

                    # Evaluación
                    eval_rewards = evaluate_agent(agent, env, EVAL_EPISODES, T_MAX)
                    tasa_exito = np.mean([r > 0 for r in eval_rewards])*100
                    
                    resultados.append({"Mapa": map_name, "Success_Rate": sr,"Gamma": gamma, "Reward_Goal": r_goal, "Iteraciones": iteration,"Tiempo_Entrenamiento": tiempo_train,"Tasa_Exito": tasa_exito})
                    env.close()

    # Exportar los resultados en un csv
    df = pd.DataFrame(resultados)
    df.to_csv(f"{output_dir}/mb_resultados_success_rate.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Success_Rate", y="Tasa_Exito", hue="Gamma", style="Mapa", markers=True)
    plt.title("MB: Impacto del Success Rate en la Tasa de Éxito")
    plt.xlabel("Success Rate")
    plt.ylabel("Tasa de éxito (%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mb_grafica_exito.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Reward_Goal", y="Tasa_Exito", hue="Mapa", marker="o")
    plt.title("MB: Evolución de la Tasa de Éxito vs Señal de Recompensa (1-100)")
    plt.xlabel("Señal de Recompensa en la Meta")
    plt.ylabel("Tasa de éxito (%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mb_grafica_exito_vs_recompensa.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Success_Rate", y="Iteraciones", hue="Gamma", style="Mapa", markers=True)
    plt.title("MB: Convergencia vs Success Rate")
    plt.xlabel("Success Rate")
    plt.ylabel("Iteraciones")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mb_grafica_convergencia.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Success_Rate", y="Tiempo_Entrenamiento", hue="Gamma", style="Mapa", markers=True)
    plt.title("MB: Tiempo Total de Entrenamiento vs Success Rate")
    plt.xlabel("Success Rate")
    plt.ylabel("Tiempo de Entrenamiento (s)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mb_grafica_tiempo_entrenamiento.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Mapa", y="Tiempo_Entrenamiento", hue="Gamma")
    plt.title("MB: Comparativa de Escalabilidad (Promedio por Mapa)")
    plt.xlabel("Tamaño del Mapa")
    plt.ylabel("Tiempo Medio de Entrenamiento (s)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mb_grafica_escalabilidad_tiempo.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Gamma", y="Tasa_Exito", hue="Mapa", marker="o")
    plt.title("MB: Impacto de Gamma en la Tasa de Éxito")
    plt.xlabel("Gamma")
    plt.ylabel("Tasa de éxito (%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mb_grafica_gamma.png")
    plt.close()

    print(f"Las gráficas y el csv están en '{output_dir}/'")

if __name__ == "__main__":
    run_experiments()