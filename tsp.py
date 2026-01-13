# --- INTEGRAÇÃO REAL COM LLMs (EXEMPLO OPENAI) ---
import os
import requests

def call_llm_openai(prompt, api_key=None, model="gpt-3.5-turbo"):
    """
    Chama a API da OpenAI para gerar relatório/instruções a partir do prompt.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Chave de API OpenAI não fornecida.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Você é um assistente logístico hospitalar."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Exemplo de uso:
# prompt = generate_llm_prompt(best_routes, delivery_points)
# resposta = call_llm_openai(prompt, api_key="SUA_CHAVE_AQUI")
# --- HEURÍSTICA SIMPLES: VIZINHO MAIS PRÓXIMO ---
def nearest_neighbor_route(cities_locations):
    if not cities_locations:
        return []
    unvisited = cities_locations[:]
    route = [unvisited.pop(0)]
    while unvisited:
        last = route[-1]
        next_city = min(unvisited, key=lambda c: calculate_distance(last, c))
        route.append(next_city)
        unvisited.remove(next_city)
    return route

# Exemplo de uso:
# nn_route = nearest_neighbor_route(cities_locations)
# nn_routes = split_routes(nn_route, N_VEHICLES)
# nn_fitness = calculate_multi_vehicle_fitness(nn_route, delivery_points, N_VEHICLES, VEHICLE_CAPACITY, VEHICLE_AUTONOMY)
# --- INTEGRAÇÃO COM LLMs PARA RELATÓRIOS E INSTRUÇÕES ---
def generate_llm_prompt(routes, delivery_points):
    """
    Gera um prompt para LLM com as rotas otimizadas e informações dos pontos de entrega.
    """
    loc_to_point = {tuple(p.location): p for p in delivery_points}
    prompt = "Relatório de Rotas para Entrega de Medicamentos e Insumos:\n"
    for idx, route in enumerate(routes):
        prompt += f"\nVeículo {idx+1}:\n"
        for order, loc in enumerate(route, 1):
            p = loc_to_point[tuple(loc)]
            prompt += f"  {order}. {p.name} (Prioridade: {p.priority}, Demanda: {p.demand})\n"
    prompt += "\nGere instruções claras para as equipes de entrega, destacando pontos prioritários e eventuais restrições.\n"
    return prompt

# Exemplo de uso:
# prompt = generate_llm_prompt(best_routes, delivery_points)
# resposta_llm = chamar_llm(prompt)

import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import mutate, order_crossover, generate_random_population, calculate_fitness, sort_population, default_problems, calculate_distance
from draw_functions import draw_paths, draw_plot, draw_cities
import sys
import numpy as np
from benchmark_att48 import *


# --- CONFIGURAÇÕES GERAIS ---
# pygame
WIDTH, HEIGHT = 800, 400
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450


# GA
N_CITIES = 15
POPULATION_SIZE = 100  # Tamanho da população
N_GENERATIONS = 200    # Número máximo de gerações
MUTATION_PROBABILITY = 0.5

# Veículos e restrições
N_VEHICLES = 2  # Exemplo: 2 veículos
VEHICLE_CAPACITY = 10  # Capacidade máxima de entregas por veículo
VEHICLE_AUTONOMY = 1000  # Distância máxima por rota (exemplo)

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# --- ESTRUTURA DOS PONTOS DE ENTREGA ---
# Cada ponto tem: nome, coordenada, prioridade, demanda
class DeliveryPoint:
    def __init__(self, name, location, priority=1, demand=1):
        self.name = name
        self.location = location  # (x, y)
        self.priority = priority  # 1 = normal, 2 = urgente, etc
        self.demand = demand      # quantidade de itens

# Exemplo de pontos de entrega (pode ser randomizado ou fixo)
delivery_points = [
    DeliveryPoint("Hospital Central", (500, 100), priority=2, demand=3),
    DeliveryPoint("Unidade A", (700, 200), priority=1, demand=2),
    DeliveryPoint("Unidade B", (600, 350), priority=1, demand=1),
    DeliveryPoint("Unidade C", (750, 300), priority=2, demand=2),
    DeliveryPoint("Domicílio 1", (650, 150), priority=1, demand=1),
    DeliveryPoint("Domicílio 2", (480, 320), priority=1, demand=1),
    DeliveryPoint("Unidade D", (550, 250), priority=1, demand=2),
    DeliveryPoint("Unidade E", (720, 120), priority=2, demand=1),
    DeliveryPoint("Domicílio 3", (470, 200), priority=1, demand=1),
    DeliveryPoint("Unidade F", (680, 380), priority=1, demand=2),
]

# Para compatibilidade com funções existentes:
cities_locations = [p.location for p in delivery_points]


# --- PREPARAÇÃO PARA MÚLTIPLOS VEÍCULOS ---
# Cada solução será uma lista de rotas, uma por veículo
def split_routes(solution, n_vehicles):
    """
    Divide uma solução (sequência de pontos) em n rotas para múltiplos veículos.
    (Simples: divisão igual, pode ser melhorado para respeitar capacidade/autonomia)
    """
    k, m = divmod(len(solution), n_vehicles)
    return [solution[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_vehicles)]

# --- FITNESS PARA MÚLTIPLOS VEÍCULOS E RESTRIÇÕES ---
def calculate_multi_vehicle_fitness(solution, delivery_points, n_vehicles, vehicle_capacity, vehicle_autonomy):
    """
    Calcula o fitness considerando múltiplos veículos, capacidade, autonomia e prioridades.
    Penaliza soluções inviáveis.
    """
    # Mapear localização para objeto DeliveryPoint
    loc_to_point = {tuple(p.location): p for p in delivery_points}
    # Dividir solução em rotas
    routes = split_routes(solution, n_vehicles)
    total_distance = 0
    penalty = 0
    for route in routes:
        if not route:
            continue
        # Capacidade e demanda
        total_demand = sum(loc_to_point[tuple(loc)].demand for loc in route)
        if total_demand > vehicle_capacity:
            penalty += 1000 * (total_demand - vehicle_capacity)
        # Distância
        route_distance = 0
        for i in range(len(route)):
            route_distance += calculate_distance(route[i], route[(i + 1) % len(route)])
        if route_distance > vehicle_autonomy:
            penalty += 1000 * (route_distance - vehicle_autonomy) / 100
        total_distance += route_distance
        # Prioridades: penalizar se pontos prioritários não estão no início da rota
        for idx, loc in enumerate(route):
            point = loc_to_point[tuple(loc)]
            if point.priority > 1 and idx > len(route)//2:
                penalty += 500 * (point.priority - 1)
    return total_distance + penalty

# Exemplo de uso:
# fitness = calculate_multi_vehicle_fitness(solution, delivery_points, N_VEHICLES, VEHICLE_CAPACITY, VEHICLE_AUTONOMY)

# --- INICIALIZAÇÃO DO PYGAME E POPULAÇÃO ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver - Distribuição de Medicamentos")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)

population = generate_random_population(cities_locations, POPULATION_SIZE)
best_fitness_values = []
best_solutions = []


# --- AJUSTE DOS OPERADORES GENÉTICOS PARA MÚLTIPLOS VEÍCULOS ---
# (Mantém crossover/mutação padrão, mas avalia solução como rota única e divide para veículos)

# --- VISUALIZAÇÃO DAS ROTAS POR VEÍCULO ---
def draw_vehicle_routes(screen, routes, colors, node_radius=NODE_RADIUS):
    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        if route:
            draw_paths(screen, route, color, width=3)
            for loc in route:
                pygame.draw.circle(screen, color, loc, node_radius)

# --- CORES PARA VEÍCULOS ---
VEHICLE_COLORS = [RED, BLUE, (0, 200, 0), (200, 0, 200), (255, 165, 0)]

# --- LOOP PRINCIPAL COM EVOLUÇÃO GENÉTICA E RELATÓRIO LLM ---
import time

def select_parents(population, fitness_func, k=3):
    # Torneio
    selected = []
    for _ in range(2):
        aspirants = random.sample(population, k)
        selected.append(min(aspirants, key=fitness_func))
    return selected


generation = 0
report_text = None
fitness_history = []
done = False
graph_shown = False
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r and done:
                # Gera relatório LLM para a melhor solução
                prompt = generate_llm_prompt(best_routes, delivery_points)
                try:
                    report_text = call_llm_openai(prompt)
                except Exception as e:
                    report_text = f"Erro ao chamar LLM: {e}"


    if not done:
        if generation < N_GENERATIONS:
            # Evolução genética
            fitness_func = lambda sol: calculate_multi_vehicle_fitness(sol, delivery_points, N_VEHICLES, VEHICLE_CAPACITY, VEHICLE_AUTONOMY)
            new_population = []
            elite = min(population, key=fitness_func)
            new_population.append(elite)  # elitismo
            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = select_parents(population, fitness_func)
                child = order_crossover(parent1, parent2)
                child = mutate(child, MUTATION_PROBABILITY)
                new_population.append(child)
            population = new_population
            generation += 1

            # Salva histórico do melhor fitness
            best_solution = min(population, key=fitness_func)
            best_fitness = fitness_func(best_solution)
            fitness_history.append(best_fitness)

            if generation >= N_GENERATIONS:
                done = True
        else:
            done = True

    # Gera relatório LLM automaticamente ao final das gerações
    if done and report_text is None:
        prompt = generate_llm_prompt(split_routes(min(population, key=lambda sol: calculate_multi_vehicle_fitness(sol, delivery_points, N_VEHICLES, VEHICLE_CAPACITY, VEHICLE_AUTONOMY)), N_VEHICLES), delivery_points)
        try:
            report_text = call_llm_openai(prompt)
        except Exception as e:
            report_text = f"Erro ao chamar LLM: {e}"

    screen.fill(WHITE)

    # Seleciona a melhor solução da população (algoritmo genético)
    fitness_func = lambda sol: calculate_multi_vehicle_fitness(sol, delivery_points, N_VEHICLES, VEHICLE_CAPACITY, VEHICLE_AUTONOMY)
    best_solution = min(population, key=fitness_func)
    best_routes = split_routes(best_solution, N_VEHICLES)
    best_fitness = fitness_func(best_solution)

    # Heurística do vizinho mais próximo
    nn_route = nearest_neighbor_route(cities_locations)
    nn_routes = split_routes(nn_route, N_VEHICLES)
    nn_fitness = calculate_multi_vehicle_fitness(nn_route, delivery_points, N_VEHICLES, VEHICLE_CAPACITY, VEHICLE_AUTONOMY)

    # Desenha as rotas de cada veículo (algoritmo genético)
    draw_vehicle_routes(screen, best_routes, VEHICLE_COLORS)

    # Destaca pontos prioritários
    for p in delivery_points:
        if p.priority > 1:
            pygame.draw.circle(screen, (255, 215, 0), p.location, NODE_RADIUS+4, 2)

    # Exibe comparativo de fitness na tela
    font = pygame.font.SysFont(None, 24)
    text1 = font.render(f"Genético: {best_fitness:.1f}", True, (0,0,0))
    text2 = font.render(f"Vizinho Próx.: {nn_fitness:.1f}", True, (0,0,0))
    text3 = font.render(f"Geração: {generation}/{N_GENERATIONS}", True, (0,0,0))
    text4 = font.render(f"População: {POPULATION_SIZE}", True, (0,0,0))
    screen.blit(text1, (10, 10))
    screen.blit(text2, (10, 35))
    screen.blit(text3, (10, 60))
    screen.blit(text4, (10, 85))

    # Exibe relatório LLM se gerado
    if report_text:
        lines = report_text.split('\n')
        for i, line in enumerate(lines[:10]):
            txt = font.render(line, True, (50,50,50))
            screen.blit(txt, (10, 120 + 22*i))

    # Ao final, mostra gráfico de evolução do fitness apenas uma vez
    if done and not graph_shown and len(fitness_history) > 1:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6,3))
        plt.plot(fitness_history)
        plt.title('Evolução do Melhor Fitness')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)
        graph_shown = True

    pygame.display.flip()
    clock.tick(FPS)

    generation = next(generation_counter)

    screen.fill(WHITE)

    population_fitness = [calculate_fitness(
        individual) for individual in population]

    population, population_fitness = sort_population(
        population,  population_fitness)

    best_fitness = calculate_fitness(population[0])
    best_solution = population[0]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    draw_plot(screen, list(range(len(best_fitness_values))),
              best_fitness_values, y_label="Fitness - Distance (pxls)")

    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    draw_paths(screen, best_solution, BLUE, width=3)
    draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)

    print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

    new_population = [population[0]]  # Keep the best individual: ELITISM

    while len(new_population) < POPULATION_SIZE:

        # selection
        # simple selection based on first 10 best solutions
        # parent1, parent2 = random.choices(population[:10], k=2)

        # solution based on fitness probability
        probability = 1 / np.array(population_fitness)
        parent1, parent2 = random.choices(population, weights=probability, k=2)

        # child1 = order_crossover(parent1, parent2)
        child1 = order_crossover(parent1, parent1)

        child1 = mutate(child1, MUTATION_PROBABILITY)

        new_population.append(child1)

    population = new_population

    pygame.display.flip()
    clock.tick(FPS)


# TODO: save the best individual in a file if it is better than the one saved.

# exit software
pygame.quit()
sys.exit()
