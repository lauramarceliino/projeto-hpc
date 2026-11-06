#!/bin/bash
#SBATCH --job-name=heat_seq
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=amd-512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=00:30:00

# =============================================================================
# FASE SEQUENCIAL (baseline). O que este script faz:
#   1. registra o ambiente (nó, CPU, gcc): onde o baseline foi medido
#   2. compila no próprio NPAD, com as flags do Makefile
#   3. valida (make test): sem baseline correto, nada do resto vale
#   4. mede o tempo do laço, várias repetições por tamanho, tudo guardado em CSV
#   5. guarda a "assinatura" do resultado de cada tamanho, que as versões
#      paralelas devem reproduzir
#
# Uso:  sbatch run_seq.slurm
#
# Os valores abaixo definem a carga de trabalho. TODAS as versões do projeto devem
# usar os mesmos TAMANHOS e ITER, e mudar qualquer um deles exige repetir o baseline.
# =============================================================================
# Trabalho ~ N^2 x ITER: N cresce por raiz de 2, então o trabalho dobra a cada tamanho
# (serve à escalabilidade fraca e ao PaScal; inclui 4096, 8192 e 16384 da ata).
TAMANHOS=${TAMANHOS:-"4096 5793 8192 11585 16384"}
ITER=${ITER:-100}        # iterações por execução (fixas em todas as versões)
REPS=${REPS:-5}          # repetições por tamanho: guardamos todas (reporte mediana ou mínimo)
PIN=${PIN:-}             # opcional: PIN="taskset -c 0" fixa o processo em um núcleo

PART=${SLURM_JOB_PARTITION:-local}
JOB=${SLURM_JOB_ID:-0}
RES=resultados
CSV="$RES/resultados.csv"             # MESMO arquivo e MESMAS colunas em todas as versões
SIG="$RES/assinatura_seq_${JOB}.txt"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "sem-git")
mkdir -p "$RES"

#1. Ambiente: sem isso ninguém sabe, depois, em que máquina saíram os números
echo "===== AMBIENTE ====="
echo "data      : $(date '+%F %T')"
echo "job       : ${JOB} | partição ${PART} | nó ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "cpu       : $(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p')"
echo "núcleos   : $(nproc)"
echo "gcc       : $(gcc --version | head -n 1)"
echo "commit    : ${COMMIT}"
# Para fixar outro gcc: module load <nome em "module available"> antes da compilação.

#2. Compilação (-B recompila aqui; a linha do gcc com as flags aparece no log)
echo "===== COMPILAÇÃO ====="
make -B heat_seq compare_grids || { echo "Falha ao compilar."; exit 1; }

#3. Validação: falhar cedo, antes de gastar tempo medindo
echo "===== VALIDAÇÃO (make test) ====="
make test || { echo "VALIDAÇÃO FALHOU: abortando antes de medir."; exit 1; }

#4. Benchmark: só o tempo do laço, que o programa mede por dentro e imprime na linha CSV
echo "===== BENCHMARK (ITER=${ITER}, REPS=${REPS}) ====="
[ -f "$CSV" ] || echo "versao,N,iteracoes,processos,threads,tempo_s,rep,particao,job" > "$CSV"
echo "# baseline sequencial | job ${JOB} | partição ${PART} | ITER=${ITER} | commit ${COMMIT}" > "$SIG"

for N in $TAMANHOS; do
    for rep in $(seq 1 "$REPS"); do
        SAIDA=$($PIN ./heat_seq -n "$N" -i "$ITER") || { echo "FALHA em N=${N} rep=${rep}"; exit 1; }
        LINHA=$(echo "$SAIDA" | grep '^CSV,')
        [ -n "$LINHA" ] || { echo "Programa não imprimiu a linha CSV (N=${N})"; exit 1; }
        echo "${LINHA#CSV,},${rep},${PART},${JOB}" >> "$CSV"
        echo "N=${N} rep=${rep}: ${LINHA#CSV,}"
        #5. Assinatura (idêntica em todas as repetições; guardamos a da primeira)
        if [ "$rep" -eq 1 ]; then
            { echo "--- N=${N} ---"
              echo "$SAIDA" | grep -E '^(soma|norma_L2|minimo|maximo|centro|perto_da_borda|residuo) '
            } >> "$SIG"
        fi
    done
done

#6. Resumo desta execução (tempo do laço, em segundos)
echo "===== RESUMO ====="
echo "N       repeticoes   minimo      media       maximo"
awk -F, -v job="$JOB" '
    NR > 1 && $1 == "seq" && $9 == job {
        n = $2; t = $6
        if (!(n in cnt) || t < mn[n]) mn[n] = t
        if (!(n in cnt) || t > mx[n]) mx[n] = t
        sm[n] += t; cnt[n]++
    }
    END { for (n in cnt) printf "%-7s %-12d %-11.6f %-11.6f %-11.6f\n", n, cnt[n], mn[n], sm[n] / cnt[n], mx[n] }
' "$CSV" | sort -n
echo "Resultados em ${RES}/ (resultados.csv e $(basename "$SIG"))"
