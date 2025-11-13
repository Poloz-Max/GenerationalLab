import random
import numpy as np
import matplotlib.pyplot as plt

POP_SIZE = 100
GENERATIONS = 100
MATCHES_PER_AGENT = 30
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3
ROCK, PAPER, SCISSORS = 0, 1, 2

def play(a, b):
    """Результат гри: 1 — перемога a, -1 — перемога b, 0 — нічия."""
    if a == b:
        return 0
    if (a == ROCK and b == SCISSORS) or (a == PAPER and b == ROCK) or (a == SCISSORS and b == PAPER):
        return 1
    return -1

class Agent:
    def __init__(self, strategy=None):
        if strategy is None:
            self.strategy = np.random.dirichlet([1, 1, 1])
        else:
            self.strategy = np.array(strategy)
        self.fitness = 0.0

    def choose(self):
        return np.random.choice([ROCK, PAPER, SCISSORS], p=self.strategy)

def evaluate(population):
    """Оцінює фітнес для всіх агентів."""
    for agent in population:
        agent.fitness = 0

    for agent in population:
        opponents = random.sample(population, MATCHES_PER_AGENT)
        score = 0
        for opp in opponents:
            res = play(agent.choose(), opp.choose())
            score += res
        agent.fitness = score / MATCHES_PER_AGENT

def tournament_selection(population):
    """Повертає найкращого з випадкової підмножини."""
    subset = random.sample(population, TOURNAMENT_SIZE)
    return max(subset, key=lambda x: x.fitness)

def crossover(p1, p2):
    """Арифметичний кросовер стратегій."""
    alpha = random.random()
    child_strategy = alpha * p1.strategy + (1 - alpha) * p2.strategy
    child_strategy /= child_strategy.sum()
    return Agent(child_strategy)

def mutate(agent):
    """Невелике випадкове зміщення ймовірностей."""
    if random.random() < MUTATION_RATE:
        noise = np.random.normal(0, 0.1, 3)
        new_strategy = agent.strategy + noise
        new_strategy = np.clip(new_strategy, 0.001, 1)
        agent.strategy = new_strategy / new_strategy.sum()

def next_generation(population):
    new_pop = []
    while len(new_pop) < len(population):
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child = crossover(parent1, parent2)
        mutate(child)
        new_pop.append(child)
    return new_pop

population = [Agent() for _ in range(POP_SIZE)]
avg_fitness = []
avg_strategy = []

for gen in range(GENERATIONS):
    evaluate(population)
    best = max(population, key=lambda x: x.fitness)
    avg_fit = sum(a.fitness for a in population) / POP_SIZE
    avg_str = np.mean([a.strategy for a in population], axis=0)
    avg_fitness.append(avg_fit)
    avg_strategy.append(avg_str)

    if gen % 10 == 0 or gen == GENERATIONS - 1:
        print(f"Покоління {gen:3d}: "
              f"середній fitness = {avg_fit:.3f}, "
              f"кращий = {best.fitness:.3f}, "
              f"середня стратегія = {avg_str.round(2)}")

    population = next_generation(population)

avg_strategy = np.array(avg_strategy)
plt.figure(figsize=(8,5))
plt.plot(avg_fitness, label="Середній fitness")
plt.title("Еволюція стратегій у грі 'Камінь–Ножиці–Папір'")
plt.xlabel("Покоління")
plt.ylabel("Fitness")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(avg_strategy[:,0], label="Камінь")
plt.plot(avg_strategy[:,1], label="Папір")
plt.plot(avg_strategy[:,2], label="Ножиці")
plt.title("Середня стратегія в популяції")
plt.xlabel("Покоління")
plt.ylabel("Ймовірність")
plt.legend()
plt.show()
