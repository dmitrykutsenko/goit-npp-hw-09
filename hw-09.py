import numpy as np
import matplotlib.pyplot as plt

# import gym
try:
    import gymnasium as gym
    GYM_BACKEND = 'gymnasium'
except ImportError:
    import gym
    GYM_BACKEND = 'gym'

# Сумісність старого gym з NumPy 2.x
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

print(f'Backend: {GYM_BACKEND}')


# Допоміжна функція для рендера
def show_render(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# Створення середовища FrozenLake
env = gym.make(
    "FrozenLake-v1",
    desc=None,
    map_name="4x4",
    is_slippery=True,
    render_mode="rgb_array"
)

state = env.reset()

frame = env.render()

# Якщо render повернув (img, info)
if isinstance(frame, tuple):
    frame = frame[0]

show_render(frame)


# Перевіримо базову інформацію
print("Кількість станів:", env.observation_space.n)
print("Кількість дій:", env.action_space.n)


# 1. Value iteration

def value_iteration(env, gamma=1.0, threshold=1e-20, max_iterations=100000):
    value_table = np.zeros(env.observation_space.n)

    for i in range(max_iterations):
        updated_value_table = np.copy(value_table)

        for state in range(env.observation_space.n):
            Q_value = []

            for action in range(env.action_space.n):
                next_states_rewards = []
                for trans_prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                    next_states_rewards.append(
                        trans_prob * (reward + gamma * updated_value_table[next_state])
                    )
                Q_value.append(np.sum(next_states_rewards))

            value_table[state] = max(Q_value)

        if np.sum(np.abs(updated_value_table - value_table)) <= threshold:
            print(f"Value-iteration converged at iteration {i+1}")
            break

    return value_table


# 2. Extract policy (для обох алгоритмів)

def extract_policy(value_table, gamma=1.0):
    policy = np.zeros(env.observation_space.n)

    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)

        for action in range(env.action_space.n):
            for trans_prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                Q_table[action] += trans_prob * (reward + gamma * value_table[next_state])

        policy[state] = np.argmax(Q_table)

    return policy


# 3. Policy evaluation (compute_value_function яку нам власне і треба)

def compute_value_function(policy, gamma=1.0, threshold=1e-10):
    value_table = np.zeros(env.observation_space.n)

    while True:
        updated_value_table = np.copy(value_table)

        for state in range(env.observation_space.n):
            action = policy[state]

            value_table[state] = sum([
                trans_prob * (reward + gamma * updated_value_table[next_state])
                for trans_prob, next_state, reward, _ in env.unwrapped.P[state][action]
            ])

        if np.sum(np.abs(updated_value_table - value_table)) <= threshold:
            break

    return value_table


# 4. Policy iteration

def policy_iteration(env, gamma=1.0):
    policy = np.zeros(env.observation_space.n)

    while True:
        value_function = compute_value_function(policy, gamma)
        new_policy = extract_policy(value_function, gamma)

        if np.all(policy == new_policy):
            print("Policy-iteration converged.")
            break

        policy = new_policy

    return policy, value_function


# 5. Запуск Value iteration

optimal_value_function = value_iteration(env, gamma=1.0)
optimal_policy_vi = extract_policy(optimal_value_function, gamma=1.0)

print("\nОптимальна функція цінності (Value Iteration):")
print(optimal_value_function)

print("\nОптимальна політика (Value Iteration):")
print(optimal_policy_vi)


# 6. Запуск Policy iteration

optimal_policy_pi, optimal_value_pi = policy_iteration(env, gamma=1.0)

print("\nОптимальна функція цінності (Policy Iteration):")
print(optimal_value_pi)

print("\nОптимальна політика (Policy Iteration):")
print(optimal_policy_pi)


# 7. Візуалізація політики у вигляді стрілок руху

def print_policy(policy):
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    grid = [arrows[int(a)] for a in policy]
    for i in range(0, 16, 4):
        print(grid[i], grid[i+1], grid[i+2], grid[i+3])


print("\nПолітика (Value Iteration):")
print_policy(optimal_policy_vi)

print("\nПолітика (Policy Iteration):")
print_policy(optimal_policy_pi)


for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()
    if isinstance(frame, tuple):
        frame = frame[0]

    print(f"Крок {step}, дія {action}, стан {obs}, винагорода {reward}")
    show_render(frame)

    if terminated or truncated:
        break

