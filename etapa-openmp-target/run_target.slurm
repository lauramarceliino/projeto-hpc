#!/bin/bash
#SBATCH --job-name=heat_target
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpu-8-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# A partição e o módulo abaixo refletem o diagnóstico histórico da ATVD13.
# Confirme a configuração atual; opções de sbatch podem sobrescrever a partição.
set -euo pipefail
export LC_ALL=C

MODULE=${MODULE:-compilers/nvidia/nvhpc/24.11}
SIZES=${SIZES:-"4096 5793 8192 11585 16384"}
ITER=${ITER:-100}
REPS=${REPS:-5}
JOB=${SLURM_JOB_ID:-0}
PART=${SLURM_JOB_PARTITION:-local}
NODE=$(hostname)
ROOT=$(pwd)
RES="$ROOT/resultados/job_${JOB}"
LOG="$RES/logs"
CSV="$RES/resultados.csv"
DET="$RES/detalhes.csv"
OUT="$ROOT/${JOB}.out"
ERR="$ROOT/${JOB}.err"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "sem-git")

mkdir -p "$LOG"
SOURCE="$RES/source"
mkdir -p "$SOURCE/etapa-openmp-target" "$SOURCE/etapa-sequencial"
guardar_logs() {
    [ ! -f "$OUT" ] || cp "$OUT" "$LOG/slurm.out"
    [ ! -f "$ERR" ] || cp "$ERR" "$LOG/slurm.err"
}
trap guardar_logs EXIT

die() { echo "ERRO: $*" >&2; exit 1; }

module load "$MODULE" || die "não foi possível carregar MODULE=$MODULE"

# Preserve o código e a configuração exatos usados por esta campanha.
cp heat_target.c Makefile run_target.slurm README.md PLANO.md analise.py graficos.py \
   "$SOURCE/etapa-openmp-target/"
cp ../etapa-sequencial/heat_common.h ../etapa-sequencial/heat_seq.c \
   ../etapa-sequencial/compare_grids.c "$SOURCE/etapa-sequencial/"
sha256sum "$SOURCE"/etapa-openmp-target/* "$SOURCE"/etapa-sequencial/* > "$LOG/SHA256SUMS.txt"

echo "===== AMBIENTE ====="
{
    echo "data=$(date '+%F %T %Z')"
    echo "job=$JOB particao=$PART node=$NODE commit=$COMMIT"
    echo "cpus=$(nproc 2>/dev/null || echo desconhecido)"
    lscpu 2>/dev/null || true
    echo "===== GPU ====="
    nvidia-smi -L 2>&1 || true
    nvidia-smi 2>&1 || true
    echo "===== MODULOS ====="
    module list 2>&1 || true
    echo "===== COMPILADOR ====="
    nvc --version 2>&1 || true
    echo "===== FLAGS ====="
    make PROFILE=nvhpc info 2>&1 || true
} > "$LOG/ambiente.txt"
cat "$LOG/ambiente.txt"

make PROFILE=nvhpc clean all > "$LOG/build.log" 2>&1 || {
    cat "$LOG/build.log"
    die "compilação falhou"
}
export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_NUM_THREADS=1
make PROFILE=nvhpc test-device > "$LOG/validacao.log" 2>&1 || {
    cat "$LOG/validacao.log"
    die "validação do dispositivo falhou; campanha cancelada"
}

echo "===== BENCHMARK ====="
echo "versao,N,iteracoes,processos,threads,tempo_s,rep,particao,job" > "$CSV"
echo "versao,N,iteracoes,tempo_total_s,tempo_kernel_s,tempo_verificacao_s,tempo_entrada_s,tempo_saida_s,bytes_h2d_estimados,bytes_d2h_estimados,offload_real,dispositivos,rep,particao,no,job,device" > "$DET"
echo "# OpenMP Target | job=$JOB | particao=$PART | no=$NODE | iter=$ITER | reps=$REPS | commit=$COMMIT" > "$RES/README.txt"
echo "# aquecimento=1 execucao descartada por versao e tamanho" >> "$RES/README.txt"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ',' || true)
GPU=${GPU:-desconhecida}

for N in $SIZES; do
    echo "===== AQUECIMENTO N=$N (resultado descartado) ====="
    warm_seq=$(./build/heat_seq -n "$N" -i "$ITER") || die "aquecimento sequencial falhou em N=$N"
    warm_target=$(./build/heat_target -n "$N" -i "$ITER") || die "aquecimento Target falhou em N=$N"
    warm_seq_sig=$(printf '%s\n' "$warm_seq" | grep -E '^(soma|norma_L2|minimo|maximo|centro|perto_da_borda|residuo) ')
    warm_target_sig=$(printf '%s\n' "$warm_target" | grep -E '^(soma|norma_L2|minimo|maximo|centro|perto_da_borda|residuo) ')
    [ -n "$warm_seq_sig" ] && [ "$warm_seq_sig" = "$warm_target_sig" ] || die "aquecimento divergiu em N=$N"
    {
        echo "===== SEQUENCIAL N=$N ====="
        printf '%s\n' "$warm_seq"
        echo "===== OPENMP TARGET N=$N ====="
        printf '%s\n' "$warm_target"
    } > "$LOG/aquecimento_N${N}.txt"

    for rep in $(seq 1 "$REPS"); do
        # Alternar a ordem diminui o viés de aquecimento e de carga do nó.
        if [ $((rep % 2)) -eq 1 ]; then
            seq_out=$(./build/heat_seq -n "$N" -i "$ITER") || die "sequencial falhou em N=$N rep=$rep"
            gpu_out=$(./build/heat_target -n "$N" -i "$ITER") || die "target falhou em N=$N rep=$rep"
        else
            gpu_out=$(./build/heat_target -n "$N" -i "$ITER") || die "target falhou em N=$N rep=$rep"
            seq_out=$(./build/heat_seq -n "$N" -i "$ITER") || die "sequencial falhou em N=$N rep=$rep"
        fi

        seq_sig=$(printf '%s\n' "$seq_out" | grep -E '^(soma|norma_L2|minimo|maximo|centro|perto_da_borda|residuo) ')
        gpu_sig=$(printf '%s\n' "$gpu_out" | grep -E '^(soma|norma_L2|minimo|maximo|centro|perto_da_borda|residuo) ')
        [ "$seq_sig" = "$gpu_sig" ] || die "assinatura divergiu em N=$N rep=$rep"

        seq_line=$(printf '%s\n' "$seq_out" | grep '^CSV,' || true)
        gpu_line=$(printf '%s\n' "$gpu_out" | grep '^CSV,' || true)
        detail_line=$(printf '%s\n' "$gpu_out" | grep '^DET,' || true)
        [ -n "$seq_line" ] && [ -n "$gpu_line" ] && [ -n "$detail_line" ] || die "saída CSV ausente"
        printf '%s,%s,%s,%s\n' "${seq_line#CSV,}" "$rep" "$PART" "$JOB" >> "$CSV"
        printf '%s,%s,%s,%s\n' "${gpu_line#CSV,}" "$rep" "$PART" "$JOB" >> "$CSV"

        device_count=$(printf '%s\n' "$detail_line" | cut -d, -f13)
        [ "$(printf '%s\n' "$detail_line" | cut -d, -f12)" = 1 ] || die "offload não foi confirmado"
        [ "$device_count" -gt 0 ] || die "runtime não reportou dispositivo"
        printf '%s,%s,%s,%s,%s,%s\n' "${detail_line#DET,}" "$rep" "$PART" "$NODE" "$JOB" "$GPU" >> "$DET"
        echo "N=$N rep=$rep | seq=$(printf '%s\n' "$seq_line" | cut -d, -f7)s | target=$(printf '%s\n' "$gpu_line" | cut -d, -f7)s"
    done
done

echo "Campanha concluída: $RES"
echo "CSV: $CSV"
echo "Detalhes: $DET"
