# Resultados — OpenMP Target

Os tempos Target abaixo só incluem observações com `offload_real=1`. O speedup usa o sequencial do mesmo job quando disponível; caso contrário, a coluna identifica o baseline histórico `amd-512`.

## GPU e referência sequencial

| N | iterações | reps | seq mediana (s) | origem seq | Target total mediana (s) | dp (s) | kernel (s) | entrada (s) | saída (s) | speedup | híbrido ref. (s; PxT) |
|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|:---|
| 4096 | 100 | 5 | 2.884384 | mesmo job/nó | 0.088550 | 0.000084 | 0.037265 | 0.033314 | 0.018017 | 32.57x | 0.0324; 1x128 (amd-512) |
| 5793 | 100 | 5 | 5.776120 | mesmo job/nó | 0.172696 | 0.000386 | 0.071813 | 0.065067 | 0.035913 | 33.45x | 0.0488; 1x128 (amd-512) |
| 8192 | 100 | 5 | 12.272934 | mesmo job/nó | 0.346749 | 0.000551 | 0.145609 | 0.129621 | 0.071624 | 35.39x | 0.3112; 2x64 (amd-512) |
| 11585 | 100 | 5 | 28.039185 | mesmo job/nó | 0.686744 | 0.000525 | 0.283548 | 0.259428 | 0.143919 | 40.83x | 0.8737; 1x128 (amd-512) |
| 16384 | 100 | 5 | 57.171875 | mesmo job/nó | 1.384493 | 0.001181 | 0.580163 | 0.517250 | 0.287018 | 41.29x | 2.0287; 1x128 (amd-512) |

`tempo_total` inclui mapeamento, kernels, verificações opcionais e leitura final. Entrada e saída são tempos agregados do runtime, não medição pura de PCIe. O tempo híbrido é referência histórica em hardware AMD e não deve ser tratado como comparação controlada com a GPU.

## Transferências estimadas

| N | bytes H2D | bytes D2H | dispositivo(s) | partição/job |
|---:|---:|---:|:---|:---|
| 4096 | 268435856 | 134218128 | Tesla V100-SXM2-16GB | gpu-8-v100/2292359 |
| 5793 | 536941984 | 268471192 | Tesla V100-SXM2-16GB | gpu-8-v100/2292359 |
| 8192 | 1073742224 | 536871312 | Tesla V100-SXM2-16GB | gpu-8-v100/2292359 |
| 11585 | 2147396000 | 1073698200 | Tesla V100-SXM2-16GB | gpu-8-v100/2292359 |
| 16384 | 4294967696 | 2147484048 | Tesla V100-SXM2-16GB | gpu-8-v100/2292359 |

## Escalabilidade forte e fraca

A tabela de tempo e speedup mantém o mesmo tamanho de problema por linha; a versão OpenMP Target usa uma GPU fixa. As variações de tamanho não são chamadas de escalabilidade fraca da GPU. Eficiência paralela `S/P` aplica-se às configurações CPU/MPI+OpenMP com `P` núcleos; a GPU é descrita por speedup, atualizações/s e custos de dados.

### Híbrido: escala forte (referência histórica amd-512, N=16384)

| núcleos | PxT | tempo (s) | speedup | eficiência | comunicação | desbalanceamento |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1x1 | 20.3436 | 1.02x | 101.5% | 0.0% | 1.000 |
| 2 | 1x2 | 16.1178 | 1.28x | 64.1% | 0.0% | 1.000 |
| 4 | 1x4 | 16.3913 | 1.26x | 31.5% | 0.0% | 1.000 |
| 8 | 1x8 | 16.9904 | 1.22x | 15.2% | 0.0% | 1.000 |
| 16 | 1x16 | 16.0112 | 1.29x | 8.1% | 0.0% | 1.000 |
| 32 | 1x32 | 8.0123 | 2.58x | 8.1% | 0.0% | 1.000 |
| 64 | 1x64 | 4.0198 | 5.14x | 8.0% | 0.0% | 1.000 |
| 128 | 2x64 | 2.0393 | 10.13x | 7.9% | 0.8% | 1.000 |
| 128 | 128x1 | 2.0654 | 10.00x | 7.8% | 19.1% | 1.209 |
| 128 | 64x2 | 2.0569 | 10.04x | 7.8% | 18.7% | 1.202 |
| 128 | 32x4 | 2.0514 | 10.07x | 7.9% | 14.8% | 1.147 |
| 128 | 16x8 | 2.0600 | 10.02x | 7.8% | 9.2% | 1.085 |
| 128 | 8x16 | 2.0521 | 10.06x | 7.9% | 1.8% | 1.002 |
| 128 | 4x32 | 2.0593 | 10.03x | 7.8% | 1.9% | 1.001 |
| 128 | 1x128 | 2.0287 | 10.18x | 8.0% | 0.0% | 1.000 |

A tabela híbrida abaixo reproduz a escalabilidade fraca com afinidade espalhada registrada na etapa MPI+OpenMP. Ela é mantida separada por usar resultados e hardware próprios.

| núcleos | N | PxT | tempo (s) | eficiência fraca | comunicação |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 1x1 | 1.1381 | 100.0% | 0.0% |
| 2 | 5793 | 1x2 | 1.2679 | 89.8% | 0.0% |
| 2 | 5793 | 2x1 | 1.1389 | 99.9% | 1.0% |
| 4 | 8192 | 1x4 | 1.1958 | 95.2% | 0.0% |
| 4 | 8192 | 2x2 | 1.1460 | 99.3% | 0.8% |
| 4 | 8192 | 4x1 | 1.1697 | 97.3% | 2.3% |
| 8 | 11585 | 1x8 | 1.1948 | 95.3% | 0.0% |
| 8 | 11585 | 8x1 | 1.2673 | 89.8% | 6.3% |
| 16 | 16384 | 1x16 | 2.0107 | 56.6% | 0.0% |
| 16 | 16384 | 8x2 | 2.0155 | 56.5% | 1.9% |
| 16 | 16384 | 16x1 | 2.0282 | 56.1% | 3.2% |
| 32 | 23170 | 1x32 | 3.8812 | 29.3% | 0.0% |
| 32 | 23170 | 8x4 | 3.9254 | 29.0% | 1.2% |
| 32 | 23170 | 32x1 | 3.9206 | 29.0% | 5.3% |
| 64 | 32768 | 1x64 | 7.8942 | 14.4% | 0.0% |
| 64 | 32768 | 8x8 | 7.9464 | 14.3% | 0.8% |
| 64 | 32768 | 64x1 | 7.9415 | 14.3% | 15.8% |
| 128 | 46341 | 1x128 | 16.6727 | 6.8% | 0.0% |
| 128 | 46341 | 8x16 | 16.7105 | 6.8% | 1.2% |
| 128 | 46341 | 128x1 | 16.4574 | 6.9% | 20.9% |

## Comunicação MPI e balanceamento

A comunicação e o desbalanceamento são métricas da implementação híbrida; as transferências H2D/D2H são métricas distintas da versão GPU. Consulte `../etapa-hibrida/analise.py` e `../etapa-hibrida/resultados/tabelas.md` para a análise completa dessas medidas.
