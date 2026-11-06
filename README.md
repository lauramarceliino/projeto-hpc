# Projeto HPC — Simulação de Difusão de Calor em uma Placa 2D

## 1. Sobre o projeto

Este projeto tem como objetivo desenvolver e avaliar diferentes estratégias de paralelização para a simulação da difusão de calor em uma placa bidimensional, utilizando o método de diferenças finitas (*Finite Difference Method*) com stencil de 5 pontos.

A simulação representa a evolução temporal da temperatura em uma matriz 2D, na qual a temperatura de cada célula é atualizada a partir da média dos seus quatro vizinhos (acima, abaixo, esquerda e direita).

O projeto busca comparar o desempenho de diferentes abordagens computacionais, desde a implementação sequencial até técnicas de paralelização em CPU e GPU.

## 2. Objetivos

- Implementar uma versão sequencial para servir como baseline de desempenho.
- Desenvolver uma implementação híbrida utilizando MPI + OpenMP.
- Desenvolver uma implementação utilizando OpenMP Target para execução em GPU.
- Desenvolver uma implementação utilizando CUDA.
- Avaliar o desempenho das diferentes abordagens.
- Analisar speedup, eficiência e escalabilidade das implementações.

## 3. Fundamentação do problema

A placa é representada por uma matriz bidimensional de valores do tipo `double`, na qual cada elemento representa a temperatura de um ponto.

A atualização ocorre iterativamente, considerando a média dos quatro vizinhos de cada célula:

```text
           cima
             |
esquerda -- célula -- direita
             |
           baixo
```

São utilizadas condições de contorno fixas, como uma das bordas mantida a 100 °C e as demais a 0 °C.

O problema apresenta características adequadas ao modelo BSP (*Bulk Synchronous Parallel*), permitindo a divisão do domínio computacional em regiões independentes, com sincronização entre as etapas de processamento.

## 4. Tecnologias utilizadas

- **Linguagem:** C/C++
- **Compilação:** GCC e Make
- **Paralelismo distribuído:** MPI
- **Paralelismo em CPU:** OpenMP
- **Computação em GPU:** OpenMP Target e CUDA
- **Ambiente de execução:** NPAD/UFRN
- **Controle de versão:** Git e GitHub

## 5. Organização do projeto

```text
projeto-hpc/
│
├── etapa-sequencial/
│   ├── Makefile
│   ├── heat_seq.c
│   ├── heat_common.h
│   ├── compare_grids.c
│   ├── run_seq.slurm
│   └── resultados/
│       ├── resultados.csv
│       └── assinaturas e logs
│
├── etapa-hibrida/             # Planejada
│   └── MPI + OpenMP
│
├── etapa-openmp-target/       # Planejada
│   └── OpenMP Target
│
├── etapa-cuda/                # Planejada
│   └── CUDA
│
├── README.md
└── .gitignore
```

## 6. Implementações

### 6.1. Versão sequencial — Baseline

Implementação inicial utilizada como referência para as comparações de desempenho.

**Status:** Concluída.

Características:
- Processamento sequencial da matriz.
- Atualização iterativa utilizando stencil de 5 pontos.
- Condições de contorno fixas.
- Validação dos resultados.
- Coleta de tempos de execução.

### 6.2. Versão híbrida — MPI + OpenMP

Implementação planejada para combinar paralelismo distribuído entre processos MPI e paralelismo por threads utilizando OpenMP.

A matriz será dividida entre os processos, com comunicação das regiões de fronteira (*halo exchange*) a cada iteração.

**Status:** Planejada.

### 6.3. Versão GPU — OpenMP Target

Implementação planejada para explorar o processamento em GPU por meio das diretivas de offloading do OpenMP.

**Status:** Planejada.

### 6.4. Versão GPU — CUDA

Implementação planejada utilizando CUDA para execução dos cálculos diretamente na GPU, explorando paralelismo em larga escala.

**Status:** Planejada.

## 7. Ambiente de execução

Os experimentos serão realizados no ambiente de computação de alto desempenho NPAD/UFRN.

A submissão dos experimentos será realizada por meio do escalonador SLURM.

Acesso ao ambiente:

```bash
ssh -p 4422 usuario@sc2.npad.ufrn.br
```

## 8. Compilação e execução

### Versão sequencial

Acessar o diretório:

```bash
cd ~/projeto-hpc/etapa-sequencial
```

Compilar:

```bash
make
```

Executar os testes de validação:

```bash
make test
```

Submeter o experimento ao SLURM:

```bash
sbatch run_seq.slurm
```

Acompanhar a execução:

```bash
squeue -u $USER
```

Consultar o log:

```bash
tail -f <ID>.out
```

## 9. Experimentos e avaliação de desempenho

Os experimentos buscam avaliar o comportamento das implementações considerando diferentes dimensões de matrizes e configurações de processamento.

As métricas analisadas serão:

- **Tempo de execução:** tempo necessário para concluir a simulação.
- **Speedup:** relação entre o tempo sequencial e o tempo paralelo.
- **Eficiência:** aproveitamento dos recursos computacionais utilizados.
- **Escalabilidade:** comportamento do desempenho conforme o aumento dos recursos ou do volume de dados.

Também serão realizadas múltiplas repetições para cada configuração, permitindo analisar a variação dos tempos de execução.

## 10. Resultados — Baseline sequencial

A primeira etapa experimental foi executada no NPAD/UFRN.

Foram utilizadas cinco dimensões de matriz, com cinco repetições para cada dimensão.

| Dimensão (N) | Mínimo (s) | Média (s) | Máximo (s) |
|---:|---:|---:|---:|
| 4096 | 1,212112 | 1,221104 | 1,226527 |
| 5793 | 2,315369 | 2,320445 | 2,325145 |
| 8192 | 4,697499 | 4,719858 | 4,742258 |
| 11585 | 9,599890 | 9,630896 | 9,658060 |
| 16384 | 20,543153 | 20,637785 | 20,716199 |

Total de medições: 25.

Os dados completos estão disponíveis em:

```text
etapa-sequencial/resultados/resultados.csv
```

O baseline sequencial será utilizado como referência para as análises comparativas das implementações paralelas.

## 11. Próximas etapas

1. Consolidar e documentar o baseline sequencial.
2. Dividir o grupo em duplas para desenvolvimento das implementações paralelas.
3. Implementar a versão híbrida MPI + OpenMP.
4. Implementar a versão OpenMP Target.
5. Implementar a versão CUDA.
6. Executar os experimentos variando dimensões, processos e threads.
7. Coletar e organizar os resultados.
8. Gerar gráficos comparativos de tempo, speedup, eficiência e escalabilidade.
9. Elaborar a análise final dos resultados.

## 12. Equipe

**Integrantes:**  
- GILDO CORDEIRO DUARTE 
- DOUGLAS FELIPE DE LIMA SILVA
- LAURA DE FATIMA MARCELINO FONSECA
- SÉRGIO LIMA SILVA DO CARMO 

**Projeto desenvolvido como parte das atividades acadêmicas da disciplina de Computação de Alto Desempenho.**
