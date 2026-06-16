import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SLIPPERY = True # Determinista o estocástico, el hielo resbala en True
MAP_NAME = "4x4" # Tamaño del mapa
CONVERGENCE_THRESHOLD = 1e-4 # Valor de convergencia para el value iteration

T_MAX = 100 # Número máximo de pasos que el agente puede dar en una partida o episodio 

def draw_rewards(rewards, title='Rewards Over Episodes'):
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True) # Esto es la cuadrícula de la gráfica
    plt.tight_layout()
    plt.show()

def print_policy(policy, map_size):
    visual_help = {0: '<', 1: 'v', 2: '>', 3: '^'}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([-1, map_size]))

class ValueIterationAgent:
    def __init__(self, env, gamma):
        self.env = env
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma

    # Es el cálculo de la ecuación de Bellman
    def calc_action_value(self, state, action):
        action_value = sum([prob*(reward + self.gamma*self.V[next_state])
                            for prob, next_state, reward, _ 
                            in self.env.unwrapped.P[state][action]])
        return action_value

    # Te dice la mejor acción que puedes hacer desde el estado actual
    def select_action(self, state):
        best_action = None
        best_value = -float('inf')
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if action_value > best_value:
                best_value = action_value
                best_action = action
        return best_action

    # Recorre las casillas para saber el cambio máximo que ha hecho, para saber si converge ya o no
    def value_iteration(self):
        max_diff = 0
        for state in range(self.env.observation_space.n):
            state_values = []
            for action in range(self.env.action_space.n):
                state_values.append(self.calc_action_value(state, action))
            new_V = max(state_values)
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
            self.V[state] = new_V
        return self.V, max_diff

    # Calcula la política óptima
    def policy(self):
        policy = np.zeros(self.env.observation_space.n)
        for s in range(self.env.observation_space.n):
            Q_values = [self.calc_action_value(s, a) for a in range(self.env.action_space.n)]
            policy[s] = np.argmax(np.array(Q_values))
        return policy

# Bucle que hace iteraciones para entrenar al agente
def train_vi(agent):
    print("Iniciando entrenamiento por Iteración de Valor...")
    max_diffs = []
    iteration = 0
    while True:
        _, max_diff = agent.value_iteration()
        max_diffs.append(max_diff)
        iteration += 1
        if max_diff < CONVERGENCE_THRESHOLD:
            print(f"¡Convergencia alcanzada en la iteración {iteration}!")
            print(f"Diferencia máxima final: {max_diff:.6f}")
            break
    return max_diffs

# Le decimos que haga x episodios, para saber su porcentaje de éxito con la política aprendida que tiene el agente
def evaluate_agent(agent, env, num_episodes, t_max):
    print(f"\nEvaluando el agente durante {num_episodes} episodios...")
    rewards = []
    exitos = 0
    for n_ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for _ in range(t_max):
            action = agent.select_action(state)
            state, reward, is_done, truncated, _ = env.step(action)
            total_reward += reward
            if is_done or truncated:
                break
        rewards.append(total_reward)
        if total_reward > 0:
            exitos += 1
    print(f"Porcentaje de éxito: {(exitos/num_episodes)*100}%")
    return rewards

# Por defecto, reward_step es 0, reward_hole es 0 y reward_goal es 1
def modify_rewards(env, reward_step, reward_hole, reward_goal):
    desc = env.unwrapped.desc.flatten()
    for state in env.unwrapped.P:
        for action in env.unwrapped.P[state]:
            new_transitions = []
            for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                letter = desc[next_state]
                if letter == b'G':
                    reward = reward_goal
                elif letter == b'H':
                    reward = reward_hole
                else:
                    reward = reward_step
                new_transitions.append((prob, next_state, reward, done))
            env.unwrapped.P[state][action] = new_transitions

if __name__ == "__main__":
    print("Configuración del Entorno") 
    
    try:
        eval_episodes = int(input("Introduce el número de episodios de evaluación [por defecto: 100]: ") or 100)
    except ValueError:
        eval_episodes = 100

    try:
        gamma = float(input("Introduce el factor de descuento (gamma) [por defecto: 0.95]: ") or 0.95)
    except ValueError:
        gamma = 0.95


    try:
        reward_goal = float(input("Introduce la recompensa por llegar a la meta [por defecto: 1.0]: ") or 1.0)
    except ValueError:
        reward_goal = 1.0
        
    try:
        reward_hole = float(input("Introduce la recompensa por caer en un hoyo [por defecto: 0.0]: ") or 0.0)
    except ValueError:
        reward_hole = 0.0
        

    try:
        reward_step = float(input("Introduce la recompensa por cada paso normal  (te cuelgan los huevos como armas) [por defecto: 0.0]: ") or 0.0)
    except ValueError:
        reward_step = 0.0

    print("\nIniciando con los parámetros introducidos..")

    env = gym.make("FrozenLake-v1", desc=None, map_name=MAP_NAME, render_mode=None, is_slippery=SLIPPERY)
    
    # Modificar las señales de recompensa del entorno
    modify_rewards(env, reward_step, reward_hole, reward_goal)
    
    agent = ValueIterationAgent(env, gamma=gamma)
    
    diffs = train_vi(agent)
    
    print("\nPolítica resultante (0:<, 1:v, 2:>, 3:^):")
    map_size = int(MAP_NAME.split('x')[0]) 
    print_policy(agent.policy(), map_size)
    
    # Obs, el value iteration no entrena por episodios, pero sí lo vamos a evaluar por episodios (para los experimentos)
    eval_rewards = evaluate_agent(agent, env, eval_episodes, T_MAX)
    
    draw_rewards(diffs, title='Convergencia de Iteración de Valor (Max Diff)')
    draw_rewards(eval_rewards, title='Recompensas en Evaluación')
    
    env.close()