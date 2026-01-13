# Sistema de Otimização de Rotas para Distribuição de Medicamentos e Insumos

## Visão Geral

Este projeto implementa um sistema de otimização de rotas baseado em Algoritmo Genético para resolver o problema de distribuição de medicamentos e insumos em ambiente hospitalar, considerando múltiplos veículos, restrições de capacidade, autonomia e prioridades de entrega. O sistema também integra uma LLM (ex: OpenAI GPT) para geração automática de relatórios e instruções para as equipes de entrega.

---

## Como Executar

1. **Pré-requisitos:**
   - Python 3.8+
   - Instale as dependências:
     ```bash
     pip install -r requirements.txt
     ```
   - Configure sua chave de API OpenAI (opcional, para relatórios LLM):
     ```bash
     export OPENAI_API_KEY="sua-chave-aqui"
     ```

2. **Execução:**
   - Execute o arquivo principal:
     ```bash
     python tsp.py
     ```
   - O sistema abrirá uma janela com a visualização das rotas otimizadas.
   - O algoritmo genético roda por 200 gerações (ajustável em `tsp.py`).
   - Ao final, o relatório LLM é gerado automaticamente e exibido na tela.

---

## Parâmetros Principais
- **POPULATION_SIZE**: Tamanho da população (padrão: 100)
- **N_GENERATIONS**: Número de gerações (padrão: 200)
- **N_VEHICLES**: Número de veículos
- **VEHICLE_CAPACITY**: Capacidade máxima de entregas por veículo
- **VEHICLE_AUTONOMY**: Distância máxima por rota
- **Prioridades**: Cada ponto de entrega pode ter prioridade (1 = normal, 2 = urgente, ...)

Estes parâmetros podem ser ajustados diretamente no início do arquivo `tsp.py`.

---

## Funcionalidades
- **Algoritmo Genético**: Evolução automática das rotas, com elitismo, crossover e mutação.
- **Restrições Realistas**: Capacidade, autonomia, múltiplos veículos, prioridades de entrega.
- **Visualização**: Rotas de cada veículo exibidas em cores distintas, destaque para entregas prioritárias.
- **Comparativo**: Exibe o fitness do algoritmo genético e da heurística do vizinho mais próximo.
- **Gráfico**: Mostra a evolução do melhor fitness ao final das gerações.
- **Relatório LLM**: Geração automática de relatório detalhado das rotas e instruções para equipes, usando LLM (ex: OpenAI GPT).

---

## Exemplo de Uso
- Ao rodar o sistema, aguarde o término das gerações.
- O relatório será exibido automaticamente na tela.
- Para ajustar parâmetros, edite o início do arquivo `tsp.py`.

---

## Estrutura dos Arquivos
- `tsp.py`: Script principal, lógica do algoritmo genético, visualização e integração LLM.
- `genetic_algorithm.py`: Funções auxiliares do algoritmo genético.
- `draw_functions.py`: Funções de visualização.
- `benchmark_att48.py`: Dados de benchmark.
- `requirements.txt`: Dependências do projeto.

---

## Observações
- O sistema foi projetado para ser facilmente adaptável a outros cenários logísticos.
- Para rodar a integração LLM, é necessário ter uma chave de API válida e acesso à internet.
- O gráfico pode não aparecer em ambientes sem suporte gráfico (ex: alguns terminais Windows). Use VS Code, Anaconda Prompt ou Jupyter para melhor experiência.

---

## Demonstração
- Recomenda-se gravar um vídeo mostrando:
  - Execução do sistema
  - Visualização das rotas
  - Comparativo de desempenho
  - Geração do relatório LLM

---

## Contato
Dúvidas ou sugestões: [Seu Nome] - [Seu Email]
