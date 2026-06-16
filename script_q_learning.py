import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
import random


# Configuración experimentos
MAP_SIZES = ["4x4", "8x8"]
SUCCESS_RATES = np.linspace(0.1, 0.95, 10)

EVAL_EPISODES = 1000
TRAIN_EPISODES = 20000
T_MAX = 100
GAMMA_DEFAULT = 0.95
LEARNING_RATE_DEFAULT = 0.5
EPSILON_DEFAULT = 0.8


def modify_env(env, success_rate, goal_reward):
    slip_rate = (1.0 - success_rate) / 2.0

    for state in env.unwrapped.P:
        for action in env.unwrapped.P[state]:
            transitions = env.unwrapped.P[state][action]
            new_transitions = []

            for i, (prob, next_state, reward, done) in enumerate(transitions):
                if len(transitions) == 1:
                    new_prob = 1.0
                elif i == 1:
                    new_prob = success_rate
                else:
                    new_prob = slip_rate

                new_reward = goal_reward if reward > 0 else reward
                new_transitions.append((new_prob, next_state, new_reward, done))

            env.unwrapped.P[state][action] = new_transitions


class QLearningAgent:
    def __init__(self, env, gamma, learning_rate, epsilon, t_max,
                 epsilon_decay=0.999, learning_rate_decay=0.9995, epsilon_min=0.01):
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.t_max = t_max
        self.epsilon_decay = epsilon_decay
        self.learning_rate_decay = learning_rate_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return np.random.choice(self.env.action_space.n)
        return np.argmax(self.Q[state, :])

    def update_Q(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state, :])
        td_target = reward + self.gamma*self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error

    def learn_from_episode(self):
        state, _ = self.env.reset()
        total_reward = 0.0

        for _ in range(self.t_max):
            action = self.select_action(state, training=True)
            new_state, new_reward, is_done, truncated, _ = self.env.step(action)
            total_reward += new_reward
            self.update_Q(state, action, new_reward, new_state)

            if is_done or truncated:
                break

            state = new_state

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.learning_rate *= self.learning_rate_decay

        return total_reward

    def policy(self):
        policy = np.zeros(self.env.observation_space.n)
        for state in range(self.env.observation_space.n):
            policy[state] = np.argmax(self.Q[state])
        return policy


def evaluate_agent(agent, env, num_episodes, t_max):
    rewards = []
    for _ in range(num_episodes):
        total_reward = 0.0
        state, _ = env.reset()

        for _ in range(t_max):
            action = agent.select_action(state, training=False)
            state, reward, is_done, truncated, _ = env.step(action)
            total_reward += reward

            if is_done or truncated:
                break

        rewards.append(total_reward)

    return rewards


def print_policy(policy, map_size):
    visual_help = {0: '<', 1: 'v', 2: '>', 3: '^'}
    policy_arrows = [visual_help[int(action)] for action in policy]
    print(np.array(policy_arrows).reshape([-1, map_size]))


def run_experiments():
    output_dir = "ql_resultados"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        gamma_input = input("Introduce los valores de Gamma, separálos por comas [por defecto: 0.95]: ")
        gammas = [float(g.strip()) for g in gamma_input.split(",")] if gamma_input.strip() else [GAMMA_DEFAULT]

        epsilon_input = input("Introduce los valores de Epsilon, separálos por comas [por defecto: 0.8]: ")
        if epsilon_input.strip():
            epsilons = [float(e.strip()) for e in epsilon_input.split(",")]
        else:
            epsilons = [EPSILON_DEFAULT]

        learning_rate_input = input("Introduce los valores de Learning Rate, separálos por comas [por defecto: 0.5]: ")
        if learning_rate_input.strip():
            learning_rates = [float(lr.strip()) for lr in learning_rate_input.split(",")]
        else:
            learning_rates = [LEARNING_RATE_DEFAULT]

        reward_input = input("Introduce los valores de señal de recompensa (Goal), sepáralos por comas [por defecto del 1 al 100 en 10 pasos]: ")
        if reward_input.strip():
            rewards_goal = [float(r.strip()) for r in reward_input.split(",")]
        else:
            rewards_goal = np.linspace(1, 100, 10)
    except ValueError:
        gammas = [GAMMA_DEFAULT]
        epsilons = [EPSILON_DEFAULT]
        learning_rates = [LEARNING_RATE_DEFAULT]
        rewards_goal = np.linspace(1, 100, 10)

    resultados = []
    total_exp = len(MAP_SIZES)*len(SUCCESS_RATES)*len(gammas)*len(epsilons)*len(learning_rates)*len(rewards_goal)
    print(f"Ejecutando {total_exp} experimentos para Q-Learning..\n")

    for r_goal in rewards_goal:
        for gamma in gammas:
            for epsilon in epsilons:
                for learning_rate in learning_rates:
                    for map_name in MAP_SIZES:
                        for sr in SUCCESS_RATES:
                            print(
                                f"Mapa={map_name} | SR={sr:.2f} | Gamma={gamma} | Epsilon={epsilon} | LR={learning_rate} | Reward={r_goal:.1f}"
                            )

                            env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=True)
                            modify_env(env, sr, r_goal)

                            agent = QLearningAgent(
                                env,
                                gamma=gamma,
                                learning_rate=learning_rate,
                                epsilon=epsilon,
                                t_max=T_MAX,
                                epsilon_decay=0.9998,
                                learning_rate_decay=0.9999,
                                epsilon_min=0.01
                            )

                            inicio_train = time.time()
                            train_rewards = []
                            for _ in range(TRAIN_EPISODES):
                                train_rewards.append(agent.learn_from_episode())
                            tiempo_train = time.time() - inicio_train

                            eval_rewards = evaluate_agent(agent, env, EVAL_EPISODES, T_MAX)
                            tasa_exito = np.mean([r > 0 for r in eval_rewards])*100

                            resultados.append(
                                {
                                    "Mapa": map_name,
                                    "Success_Rate": sr,
                                    "Gamma": gamma,
                                    "Epsilon": epsilon,
                                    "Learning_Rate": learning_rate,
                                    "Reward_Goal": r_goal,
                                    "Iteraciones": TRAIN_EPISODES,
                                    "Tiempo_Entrenamiento": tiempo_train,
                                    "Tasa_Exito": tasa_exito,
                                    "Reward_Medio_Entrenamiento": float(np.mean(train_rewards)),
                                }
                            )
                            env.close()

    df = pd.DataFrame(resultados)
    df.to_csv(f"{output_dir}/ql_resultados_success_rate.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Success_Rate", y="Tasa_Exito", hue="Mapa", marker="o")
    plt.title("QL: Impacto del Success Rate en la Tasa de Éxito")
    plt.xlabel("Success Rate")
    plt.ylabel("Tasa de éxito (%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ql_grafica_exito.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Reward_Goal", y="Tasa_Exito", hue="Mapa", marker="o")
    plt.title("QL: Evolución de la Tasa de Éxito vs Señal de Recompensa (1-100)")
    plt.xlabel("Señal de Recompensa en la Meta")
    plt.ylabel("Tasa de éxito (%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ql_grafica_exito_vs_recompensa.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Success_Rate", y="Tiempo_Entrenamiento", hue="Learning_Rate", style="Mapa", markers=True)
    plt.title("QL: Tiempo Total de Entrenamiento vs Success Rate (por Learning Rate)")
    plt.xlabel("Success Rate")
    plt.ylabel("Tiempo de Entrenamiento (s)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ql_grafica_tiempo_vs_success_learning_rate.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Success_Rate", y="Tiempo_Entrenamiento", hue="Gamma", style="Mapa", markers=True)
    plt.title("QL: Tiempo Total de Entrenamiento vs Success Rate")
    plt.xlabel("Success Rate")
    plt.ylabel("Tiempo de Entrenamiento (s)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ql_grafica_tiempo_entrenamiento.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Gamma", y="Tasa_Exito", hue="Mapa", marker="o")
    plt.title("QL: Impacto de Gamma en la Tasa de Éxito")
    plt.xlabel("Gamma")
    plt.ylabel("Tasa de éxito (%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ql_grafica_gamma.png")
    plt.close()

    print(f"Las gráficas y el csv están en '{output_dir}/'")


if __name__ == "__main__":
    run_experiments()