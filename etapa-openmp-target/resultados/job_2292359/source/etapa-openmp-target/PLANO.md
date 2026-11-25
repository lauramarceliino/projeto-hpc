# Plano de implementação — OpenMP Target

## Objetivo e limites

Este documento registra as decisões para a próxima etapa do projeto: implementar e avaliar, em **C**, a difusão de calor 2D com offloading OpenMP Target para GPU. A nova etapa será uma pasta irmã de `etapa-sequencial/` e `etapa-hibrida/`, preservando a organização, os formatos de resultado e o estilo conciso de código já adotados.

O plano cobre uma primeira versão correta e mensurável para **uma GPU**. Não inclui CUDA, MPI na GPU, multi-GPU ou política adaptativa CPU/GPU na primeira implementação. Qualquer extensão desse tipo deve ser documentada e medida separadamente.

## Estrutura prevista

```text
etapa-openmp-target/
├── PLANO.md                 # este registro de decisões
├── README.md                # comandos e operação da etapa
├── heat_target.c            # implementação C/OpenMP Target
├── Makefile                 # perfis de compilação e validação
├── run_target.slurm         # compilação/testes/experimentos no NPAD
├── analise.py               # tabelas, speedup e eficiência aplicável
├── graficos.py              # figuras comparativas
└── resultados/
    ├── job_<id>/
    │   ├── logs/             # ambiente, build, validação, aquecimento e execução
    │   ├── source/           # fontes exatas, preservando os diretórios relativos
    │   │   ├── etapa-openmp-target/
    │   │   └── etapa-sequencial/
    │   ├── resultados.csv    # tempo total por repetição (seq e Target)
    │   └── detalhes.csv      # tempos internos e bytes estimados do Target
    ├── graficos/
    └── tabelas.md
```

Manter a convenção de um diretório por job e preservar dados brutos, logs e metadados. Os resultados gráficos e tabelas usados na análise final devem permanecer reproduzíveis a partir dos CSVs.

## Contratos que a implementação deve preservar

- Usar C11 e as diretivas OpenMP Target; não converter a implementação para C++ nem chamar CUDA diretamente.
- Reutilizar `../etapa-sequencial/heat_common.h` para alocação, inicialização determinística, índices, estatísticas, resíduo e validação comum.
- Preservar a placa `N x N` em armazenamento contíguo row-major, o stencil de cinco pontos, o passo Jacobi e as bordas fixas (linha superior a 100 °C e demais bordas a 0 °C).
- Manter os mesmos parâmetros, casos de teste e condições iniciais das versões sequencial e híbrida. Preservar as opções de CLI compatíveis sempre que aplicáveis (`-n`, `-i`, `-c`, `-t`, `-o`, `-a`).
- Não alterar os resultados já publicados das etapas sequencial e híbrida. Atualizar o README principal somente depois de a nova etapa estar implementada, validada e executada.
- Documentar no próprio código as decisões relevantes: layout, troca dos buffers, mapeamento, sincronização e medição.

## Desenho do kernel e da memória

1. Inicializar no host dois buffers contíguos com as mesmas bordas fixas. Mapear ambos para a GPU antes do laço, para que as bordas permaneçam válidas durante a alternância dos buffers.
2. Manter os endereços dos dois buffers fixos e alternar origem/destino pela paridade da iteração. Isso evita depender de remapeamento de ponteiros no dispositivo.
3. Executar o stencil interior com `target teams distribute parallel for collapse(2)`. Cada célula de destino deve ter um único escritor. A região deve terminar de forma síncrona para garantir que a grade de uma iteração esteja pronta antes da próxima.
4. Manter os dados residentes durante todo o laço com uma região `target data`; não transferir grades a cada passo. Ao final, copiar de volta somente o buffer que contém o resultado final.
5. Não adicionar `nowait`, atualizações intermediárias, memória unificada ou otimizações de kernel antes da validação da versão básica. Cada otimização posterior deve ter comparação própria contra essa referência.

A semântica dos mapeamentos persistentes é definida por `target data`; `target update` pode sincronizar itens selecionados entre o host e o dispositivo. Consultar a especificação oficial ao implementar as cláusulas: [OpenMP 5.2 — target data](https://www.openmp.org/spec-html/5.2/openmpse82.html), [OpenMP 5.2 — target update](https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-2.pdf) e [OpenMP Reference Guide — construções combinadas](https://www.openmp.org/wp-content/uploads/OpenMPRefGuide-5.2-Web-2024.pdf).

## Compilação, execução e confirmação do dispositivo

Antes de fixar os parâmetros de `run_target.slurm`, verificar no NPAD a partição GPU disponível, GPUs, CPUs por tarefa, módulos, compilador e runtime OpenMP. Os registros da ATVD13 mostram uma execução anterior com NVIDIA V100 e NVHPC 24.11, mas são evidência histórica, não uma garantia de configuração atual ([diagnóstico](../../../ATVD13/docs/diagnostico_npad.txt), [README da ATVD13](../../../ATVD13/README.md)).

O job deve registrar hostname, partição, recursos Slurm, versão do compilador/runtime, flags e informações do dispositivo. O programa deve consultar `omp_is_initial_device()` no probe e registrar uma flag em cada kernel, rejeitando a execução se qualquer passo ocorreu no host. `OMP_TARGET_OFFLOAD=MANDATORY` deve ser usado como proteção adicional se for suportado pelo runtime selecionado; não substitui as verificações no programa.

O Makefile deve separar pelo menos a compilação para validação host-only da compilação com offloading GPU, sem tratar um binário host-only como evidência de execução na GPU. O modo host-only valida a fórmula e as saídas com uma implementação de referência local; só `test-device` valida o caminho de offloading. A arquitetura do dispositivo e as flags definitivas serão escolhidas com base no compilador e no nó GPU efetivamente disponíveis.

## Validação numérica

Antes de qualquer campanha de desempenho:

1. Executar os casos de ponto fixo uniforme e linear, além do caso da placa.
2. Testar tamanhos pequenos e irregulares, número de iterações baixo e ímpar, e as bordas da grade.
3. Comparar a grade final com `etapa-sequencial/heat_seq.c` e `etapa-sequencial/compare_grids.c`, além de conferir resíduo, mínimo, máximo e valores finitos.
4. Usar flags estritas, sem fast-math. Buscar igualdade exata quando a ordem aritmética e o compilador permitirem; se houver diferenças legítimas de arredondamento, documentar e aplicar tolerância absoluta/relativa explícita.
5. Registrar a validação no log e recusar incluir em CSV de desempenho uma execução que falhou ou caiu silenciosamente no host.

## Protocolo experimental

### Comparação controlada

O projeto já tem medições sequenciais em `etapa-sequencial/resultados/` e campanhas híbridas em `etapa-hibrida/resultados/`, inclusive análise detalhada de comunicação e desbalanceamento. Elas serão preservadas como referência histórica. A comparação principal deve repetir as três versões — sequencial, MPI+OpenMP e OpenMP Target — com os mesmos `N`, iterações, caso inicial, número de repetições e política de compilação tão equivalente quanto possível.

Priorizar medições no mesmo nó e ambiente, se o NPAD permitir executar as versões CPU e GPU na partição GPU com recursos suficientes. Se isso não for possível, executar as versões disponíveis e registrar explicitamente hardware, compilador e partição; não atribuir diferenças exclusivamente ao modelo de paralelismo quando também mudaram nó ou compilador. Executar e descartar um aquecimento de cada versão por tamanho antes das medições, guardar suas saídas em `logs/`, usar pelo menos cinco repetições válidas por configuração e alternar a ordem das versões para reduzir efeitos de ordem e aquecimento.

Os tempos já existentes na ATVD13 não são resultados comparáveis: aquela tarefa resolve propagação de onda, não difusão de calor. Seus scripts e registros servem apenas como referência operacional de offloading e submissão ao NPAD.

### Métricas e interpretação

- **Tempo total**: inclui o custo necessário para executar a aplicação, inclusive entrada e saída dos dados da GPU. Esta é a medida principal para speedup percebido.
- **Tempo do kernel/laço**, **entrada/mapeamento** e **leitura final**: registrar separadamente para explicar o custo. Sem profiler de hardware, identificar entrada e leitura como tempos agregados de runtime, alocação/mapeamento, sincronização e transferência, não como tráfego puro no barramento.
- **Speedup**: calcular com medianas e o sequencial correspondente à mesma malha: `S = T_seq(N) / T_versao(N)`.
- **Eficiência paralela CPU/MPI+OpenMP**: `E = S / P`, declarando `P` como o número de núcleos alocados (`processos x threads`).
- **GPU**: informar speedup usando o tempo total, atualizações de célula por segundo e, se os dados permitirem, largura de banda efetiva estimada. Não dividir speedup pelo número de GPUs e chamar isso de eficiência paralela: uma GPU é uma configuração fixa e seus recursos internos não equivalem diretamente a núcleos de CPU.
- **MPI e balanceamento**: manter os percentuais de comunicação e o desbalanceamento da etapa híbrida. Apresentar esses custos ao lado das transferências CPU–GPU, sem somá-los nem tratá-los como a mesma forma de comunicação.

### Escalabilidade forte e fraca

- **Forte**: manter `N` e iterações fixos enquanto varia o número de núcleos nas versões sequencial/híbrida. Comparar a GPU, como dispositivo único, com esses tempos e registrar o tamanho em que o custo de transferência deixa de dominar. Variações de `num_teams`/`thread_limit`, se avaliadas, serão identificadas como ajuste de lançamento do kernel, não como adição de recursos físicos.
- **Fraca CPU/híbrida**: aumentar o número de pontos da grade aproximadamente na mesma proporção do número de núcleos. Como o domínio é quadrado, escalar o lado `N` por `sqrt(P)` mantém aproximadamente o trabalho por núcleo. Usar a série e a convenção já presentes na análise híbrida.
- **Fraca GPU**: uma GPU única não permite escalabilidade fraca clássica porque a quantidade de dispositivos não aumenta. Para a primeira versão, apresentar desempenho conforme o tamanho da entrada e declarar esse limite; só incluir weak scaling próprio de GPU se houver múltiplas GPUs disponíveis e uma extensão multi-GPU explicitamente implementada.

## Dados e artefatos da campanha

`resultados.csv` deve ter uma linha por repetição válida e incluir: versão, `N`, iterações, processos, threads do host, tempo total, repetição, partição e job. `detalhes.csv` acrescenta tempo do kernel/laço, tempo de verificação, entrada, saída, bytes estimados por direção (incluindo flags de validação), confirmação de offload, número de dispositivos, nó e modelo da GPU. O job copia as fontes e configurações usadas para `source/` e registra seus SHA-256; compilador e flags ficam em `logs/ambiente.txt` e `logs/build.log`. As estatísticas devem ser recalculáveis a partir dos CSVs, com mediana e dispersão claramente identificadas.

`analise.py` deve gerar tabelas reproduzíveis e combinar os CSVs existentes sem reescrevê-los. `graficos.py` deve produzir, no mínimo, comparação de tempo total, speedup, eficiência CPU/híbrida e custos de entrada/kernel/saída da GPU. Gráficos devem identificar configurações e unidades; não extrapolar resultados para hardware não testado.

## Critérios para considerar a etapa pronta

- O código compila em C com o perfil GPU selecionado e o offloading real é confirmado no NPAD.
- Os testes numéricos passam e a comparação com a versão sequencial respeita o critério documentado.
- O job salva código/configuração usados, metadados de ambiente, logs, CSV bruto e estado de validação em diretório próprio.
- A análise é regenerável a partir dos resultados e distingue tempo total de tempos de kernel e dados.
- As comparações de escalabilidade, eficiência, comunicação MPI, balanceamento e transferência GPU respeitam as definições deste plano e declaram as limitações de hardware.
- O README do projeto é atualizado para refletir apenas o que foi efetivamente implementado e medido.
