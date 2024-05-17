from deap import base, creator, tools, algorithms
import random

# 定义优化问题
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))  # 多目标：最小化mem和cpu，最大化带宽
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 定义评价函数
def evalFunc(individual):
    mem = individual[0]
    cpu = individual[1]
    bandwidth = individual[2]
    return mem, cpu, bandwidth

toolbox.register("evaluate", evalFunc)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0/3.0)
toolbox.register("select", tools.selNSGA2)

# 运行遗传算法
population = toolbox.population(n=100)
algorithms.eaSimple(population, toolbox, cxpb=0.9, mutpb=0.1, ngen=100, verbose=False)

# 输出最优解
best_individuals = tools.selBest(population, k=1)
best_mem = best_individuals[0][0]
best_cpu = best_individuals[0][1]
best_bandwidth = best_individuals[0][2]
print("Best Individual: ", best_mem, best_cpu, best_bandwidth)
