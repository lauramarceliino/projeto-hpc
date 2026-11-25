# OpenMP Target — difusão de calor 2D

Versão em C11 do stencil Jacobi com offloading OpenMP Target para uma GPU. O
código reutiliza `etapa-sequencial/heat_common.h`, preserva as condições de
contorno e mantém os dois buffers mapeados durante a evolução temporal.

O plano de implementação, as definições das métricas e os limites de
escalabilidade estão registrados em [PLANO.md](PLANO.md).

## Compilação e testes

Teste host-only para conferir a lógica, os casos fixos e a igualdade da grade.
Esse perfil seleciona um caminho sequencial local e **não executa nem comprova
os kernels na GPU**:

```bash
make PROFILE=host test-host
```

No NPAD, confira a partição GPU e carregue o módulo correspondente. A
configuração abaixo reproduz a referência histórica da ATVD13 e deve ser
confirmada no cluster antes de submeter:

```bash
module load compilers/nvidia/nvhpc/24.11
make PROFILE=nvhpc all
OMP_TARGET_OFFLOAD=MANDATORY make PROFILE=nvhpc test-device
```

`test-device` falha se o probe ou os kernels forem executados no host. A opção
`-H` em `heat_target` seleciona o caminho host-only para testes; não a use em
benchmarks. Cada kernel grava o resultado de `omp_is_initial_device()` em uma
flag, lida no final para confirmar offloading real em todos os passos.

## Campanha no NPAD

Submeta a partir deste diretório. O job compila no nó alocado, valida o
offloading real e mede o sequencial e OpenMP Target no mesmo nó. Para cada
tamanho, uma execução de aquecimento por versão é descartada antes das
repetições medidas; a ordem das versões medidas é alternada:

```bash
sbatch run_target.slurm
```

É possível substituir o conjunto padrão de tamanhos, iterações e repetições:

```bash
SIZES="4096 8192 16384" ITER=100 REPS=5 sbatch run_target.slurm
```

O job `2292359` confirmou a configuração `gpu-8-v100`, `gpu:1` e NVHPC 24.11.
Para novas campanhas, confira a configuração vigente e passe conta ou QoS ao
`sbatch` se forem exigidas, por exemplo `sbatch --account=CONTA --qos=QOS ...`.
Os resultados ficam isolados em `resultados/job_<id>/`: CSV comum, CSV
detalhado, ambiente, compilação, validação, saídas dos aquecimentos descartados,
códigos-fonte usados na campanha com seus SHA-256 e logs Slurm. A estrutura
`source/` mantém os diretórios `etapa-openmp-target/` e `etapa-sequencial/`,
para que os caminhos de inclusão e compilação sejam reproduzíveis.

## Análise

Após copiar os resultados para a estação de análise, execute:

```bash
python3 analise.py
python3 graficos.py
```

`analise.py` gera `resultados/tabelas.md`. `graficos.py` salva figuras em
`resultados/graficos/` e requer Matplotlib. O tempo total inclui entrada,
kernels e leitura final. Os tempos de entrada/saída incluem custos do runtime
e sincronização, portanto não representam tráfego puro de barramento. A
estimativa H2D inclui as flags inicializadas para detectar kernels sem execução;
a estimativa D2H conta as flags lidas ao confirmar cada kernel.

A comparação principal usa o sequencial executado no mesmo job GPU. Os tempos
híbridos existentes em `etapa-hibrida/resultados/` são apresentados como
referência do `amd-512`; eles não são uma comparação controlada com a GPU. Uma
GPU única não oferece escalabilidade fraca clássica: a análise apresenta sua
curva por tamanho e reserva eficiência `S/P` e escala fraca para configurações
CPU/MPI+OpenMP com recursos claramente contados.

**Estado:** implementado, validado e medido no NPAD (job `2292359`, Tesla V100,
cinco tamanhos e cinco repetições por tamanho). Consulte
`resultados/tabelas.md`, `resultados/graficos/` e o diretório bruto do job.
