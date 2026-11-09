#!/bin/bash
#SBATCH --job-name=heat_hib
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=amd-512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00

# Uso: sbatch run_hib.slurm
# CONFIGS: lista PxT (processos x threads). Mesmos SIZES/ITER/REPS do baseline.
# PAIRS: lista N:PxT que substitui SIZES x CONFIGS (ex.: escalabilidade fraca)
#   PAIRS="4096:1x1 5793:2x1 8192:4x1" CSV=resultados/fraca.csv sbatch run_hib.slurm
# AFFINITY: compact (T núcleos vizinhos por rank) | spread (ranks e threads espalhados pelo nó)
# MODES: overlap (halo sobreposto) e/ou blocking (-b)
export LC_ALL=C

MODULES=${MODULES:-"libraries/openmpi/4.1.8-gnu8"}
SIZES=${SIZES:-"4096 5793 8192 11585 16384"}
ITER=${ITER:-100}
REPS=${REPS:-5}
CONFIGS=${CONFIGS:-"1x1 1x2 1x4 1x8 1x16 1x32 1x64 2x64 128x1 64x2 32x4 16x8 8x16 4x32 1x128"}
PAIRS=${PAIRS:-}
LAUNCHER=${LAUNCHER:-srun}
MODES=${MODES:-"overlap"}
AFFINITY=${AFFINITY:-compact}

PART=${SLURM_JOB_PARTITION:-local}
JOB=${SLURM_JOB_ID:-0}
RES=resultados
CSV=${CSV:-$RES/resultados.csv}
DET=${DET:-${CSV%.csv}_detalhes.csv}
SIG="$RES/assinatura_hib_${JOB}.txt"
SEQ_SIG=$(ls -t ../etapa-sequencial/resultados/assinatura_seq_*.txt 2>/dev/null | head -n 1)
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "sem-git")
NODE_CPUS=${SLURM_CPUS_ON_NODE:-256}
mkdir -p "$RES"

[ -n "$MODULES" ] && module load $MODULES
command -v mpicc >/dev/null || { echo "mpicc não encontrado: defina MODULES (module avail 2>&1 | grep -i mpi)"; exit 1; }

export OMP_PLACES=cores
if [ "$AFFINITY" = "spread" ]; then export OMP_PROC_BIND=spread; else export OMP_PROC_BIND=close; fi

# Lança P processos com T threads. O SLURM conta threads de hardware (SMT): 2 CPUs por núcleo.
launch() {
    local p=$1 t=$2; shift 2
    local cpus=$((2 * t))
    [ "$AFFINITY" = "spread" ] && cpus=$((NODE_CPUS / p))
    if [ "$LAUNCHER" = "srun" ]; then
        OMP_NUM_THREADS=$t srun --mpi=pmix --mem=0 -n "$p" -c "$cpus" --cpu-bind=cores "$@"
    else
        OMP_NUM_THREADS=$t mpirun -np "$p" --map-by "slot:PE=$t" --bind-to core -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND "$@"
    fi
}

echo "===== AMBIENTE ====="
echo "data      : $(date '+%F %T')"
echo "job       : ${JOB} | partição ${PART} | nó(s) ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "cpu       : $(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p')"
echo "núcleos   : $(nproc)"
echo "mpicc     : $(mpicc --version | head -n 1)"
echo "mpi       : $(mpirun --version 2>&1 | head -n 1)"
echo "launcher  : ${LAUNCHER} | afinidade ${AFFINITY}"
echo "commit    : ${COMMIT}"

echo "===== COMPILAÇÃO ====="
make -B heat_hib seq || { echo "Falha ao compilar."; exit 1; }

echo "===== VALIDAÇÃO (make test) ====="
if [ "$LAUNCHER" = "srun" ]; then TEST_RUN="srun --mpi=pmix"; else TEST_RUN="mpirun"; fi
make test MPIRUN="$TEST_RUN" || { echo "VALIDAÇÃO FALHOU: abortando antes de medir."; exit 1; }

echo "===== BENCHMARK (ITER=${ITER}, REPS=${REPS}) ====="
[ -f "$CSV" ] || echo "versao,N,iteracoes,processos,threads,tempo_s,rep,particao,job" > "$CSV"
[ -f "$DET" ] || echo "versao,N,iteracoes,processos,threads,tempo_s,calc_max_s,calc_med_s,comm_max_s,comm_med_s,desbalanceamento,rep,particao,job" > "$DET"
echo "# híbrido | job ${JOB} | partição ${PART} | ITER=${ITER} | commit ${COMMIT} | ref ${SEQ_SIG:-nenhuma}" > "$SIG"
mismatches=0

if [ -z "$PAIRS" ]; then
    for N in $SIZES; do for cfg in $CONFIGS; do PAIRS="$PAIRS $N:$cfg"; done; done
fi

for mode in $MODES; do
    flag=""; [ "$mode" = "blocking" ] && flag="-b"
    for pair in $PAIRS; do
        N=${pair%%:*}; cfg=${pair#*:}
        P=${cfg%x*}; T=${cfg#*x}
        ref_sig=""
        [ -n "$SEQ_SIG" ] && ref_sig=$(sed -n "/^--- N=${N} ---/,/^---/p" "$SEQ_SIG" | grep -v '^---')
        for rep in $(seq 1 "$REPS"); do
            out=$(launch "$P" "$T" ./heat_hib -n "$N" -i "$ITER" $flag) || { echo "FALHA N=${N} ${cfg} rep=${rep}"; exit 1; }
            line=$(echo "$out" | grep '^CSV,')
            [ -n "$line" ] || { echo "Sem linha CSV (N=${N} ${cfg})"; exit 1; }
            echo "${line#CSV,},${rep},${PART},${JOB}" >> "$CSV"
            echo "$(echo "$out" | grep '^DET,' | cut -d, -f2-),${rep},${PART},${JOB}" >> "$DET"
            echo "N=${N} ${cfg} ${mode} rep=${rep}: ${line#CSV,}"
            # a assinatura é a mesma em todas as repetições: confere só a primeira
            if [ "$rep" -eq 1 ]; then
                cur_sig=$(echo "$out" | grep -E '^(soma|norma_L2|minimo|maximo|centro|perto_da_borda|residuo) ')
                { echo "--- N=${N} ${cfg} ${mode} ---"; echo "$cur_sig"; } >> "$SIG"
                if [ -n "$ref_sig" ] && [ "$cur_sig" != "$ref_sig" ]; then
                    echo "  ATENÇÃO: assinatura difere do sequencial (N=${N} ${cfg})"
                    mismatches=$((mismatches + 1))
                fi
            fi
        done
    done
done

echo "===== RESUMO (tempo do laço, s) ====="
echo "versao     N       PxT        reps  minimo      media       maximo"
awk -F, -v job="$JOB" '
    NR > 1 && $9 == job {
        k = sprintf("%-10s %-7s %-10s", $1, $2, $4 "x" $5); t = $6
        if (!(k in c) || t < mn[k]) mn[k] = t
        if (!(k in c) || t > mx[k]) mx[k] = t
        s[k] += t; c[k]++
    }
    END { for (k in c) printf "%s %-5d %-11.6f %-11.6f %-11.6f\n", k, c[k], mn[k], s[k] / c[k], mx[k] }
' "$CSV" | sort -k2,2n -k1,1 -k3,3V
if [ -n "$SEQ_SIG" ]; then
    echo "assinaturas vs sequencial ($(basename "$SEQ_SIG")): ${mismatches} divergência(s)"
fi
echo "Resultados em ${RES}/ (resultados.csv e $(basename "$SIG"))"
