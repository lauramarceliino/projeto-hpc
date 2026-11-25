#!/usr/bin/env python3
"""Gera figuras de tempo, speedup, escalabilidade e transferências.

Uso: python3 graficos.py (requer matplotlib; os PNGs vão para resultados/graficos/).
"""
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analise import (HIB_DETAILS, SEQ_CSV, HIB_CSV, gpu_summary, read_csv,
                     seq_historical, seq_same_job, target_common, target_details)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "resultados" / "graficos"
INK, MUTED, GRID, BG = "#18212b", "#65717d", "#dce2e8", "#ffffff"
BLUE, GREEN, ORANGE = "#2878b5", "#22966b", "#e47b32"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
})


def save(fig, name, title):
    fig.suptitle(title, x=0.08, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def same_job_seq(common, target_rows, n, it, old_seq):
    jobs = sorted({r.get("job", "") for r in target_rows
                   if int(r.get("N", 0)) == n and int(r.get("iteracoes", 0)) == it})
    values = []
    for row in common:
        if (row.get("versao") == "seq" and int(row.get("N", 0)) == n and
                int(row.get("iteracoes", 0)) == it and row.get("job", "") in jobs):
            values.append(float(row["tempo_s"]))
    return median(values) if values else old_seq.get(n)


def plot_times(summary, target_rows, common, old_seq):
    points = []
    for (n, it), data in sorted(summary.items()):
        seq = same_job_seq(common, target_rows, n, it, old_seq)
        if seq is not None:
            points.append((n, seq, data["median"]["tempo_total_s"]))
    if not points:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = [p[0] for p in points]
    ax.plot(x, [p[1] for p in points], marker="o", lw=2, color=BLUE, label="sequencial")
    ax.plot(x, [p[2] for p in points], marker="o", lw=2, color=GREEN, label="OpenMP Target total")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(n) for n in x])
    ax.set_xlabel("lado N da grade")
    ax.set_ylabel("tempo mediano (s)")
    ax.legend()
    save(fig, "1_tempo_total.png", "Tempo total: sequencial e OpenMP Target")
    return True


def plot_speedup(summary, target_rows, common, old_seq):
    points = []
    for (n, it), data in sorted(summary.items()):
        seq = same_job_seq(common, target_rows, n, it, old_seq)
        if seq is not None and data["median"]["tempo_total_s"] > 0:
            points.append((n, seq / data["median"]["tempo_total_s"]))
    if not points:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x, y = zip(*points)
    ax.plot(x, y, marker="o", lw=2, color=ORANGE)
    ax.axhline(1, color=MUTED, ls="--", lw=1, label="mesmo tempo")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(n) for n in x])
    ax.set_xlabel("lado N da grade")
    ax.set_ylabel("speedup sobre o sequencial")
    ax.legend()
    save(fig, "2_speedup_target.png", "Speedup OpenMP Target incluindo transferência")
    return True


def plot_transfers(summary):
    items = sorted(summary.items())
    if not items:
        return False
    n = [key[0] for key, _ in items]
    parts = {
        "entrada/mapeamento": [data["median"]["tempo_entrada_s"] for _, data in items],
        "kernels": [data["median"]["tempo_kernel_s"] for _, data in items],
        "leitura final": [data["median"]["tempo_saida_s"] for _, data in items],
    }
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = [0.0] * len(n)
    for label, values, color in (("entrada/mapeamento", parts["entrada/mapeamento"], BLUE),
                                 ("kernels", parts["kernels"], GREEN),
                                 ("leitura final", parts["leitura final"], ORANGE)):
        ax.bar(range(len(n)), values, bottom=bottom, color=color, label=label, width=0.7)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(range(len(n)), [str(x) for x in n])
    ax.set_xlabel("lado N da grade")
    ax.set_ylabel("tempo mediano (s)")
    ax.legend()
    save(fig, "3_transferencias.png", "Custo agregado de dados e kernels OpenMP Target")
    return True


def plot_hybrid_strong():
    rows = [r for r in read_csv(HIB_DETAILS) if r.get("experimento") == "forte"
            and r.get("versao") == "hib" and r.get("N") == "16384"]
    seq = seq_historical().get(16384)
    if not rows or not seq:
        return False
    groups = defaultdict(list)
    for r in rows:
        groups[(int(r["processos"]) * int(r["threads"]))].append(float(r["tempo_s"]))
    cores = sorted(groups)
    speedup = [seq / median(groups[c]) for c in cores]
    efficiency = [100 * s / c for s, c in zip(speedup, cores)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(cores, efficiency, marker="o", lw=2, color=BLUE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(cores, [str(c) for c in cores])
    ax.set_xlabel("núcleos alocados (MPI × OpenMP)")
    ax.set_ylabel("eficiência paralela (%)")
    ax.set_title("Híbrido — referência histórica amd-512", loc="left", color=MUTED, fontsize=9)
    save(fig, "4_eficiencia_hibrida_forte.png", "Eficiência da escala forte híbrida (N=16384)")
    return True


def plot_hybrid_weak():
    rows = [r for r in read_csv(HIB_DETAILS) if r.get("experimento") == "fraca_espalhada"
            and r.get("versao") == "hib"]
    base_rows = [r for r in rows if r.get("N") == "4096" and r.get("processos") == "1"
                 and r.get("threads") == "1"]
    if not rows or not base_rows:
        return False
    base = median(float(r["tempo_s"]) for r in base_rows)
    groups = defaultdict(list)
    for r in rows:
        cores = int(r["processos"]) * int(r["threads"])
        groups[cores].append(base / float(r["tempo_s"]) * 100)
    cores = sorted(groups)
    efficiency = [median(groups[c]) for c in cores]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(cores, efficiency, marker="o", lw=2, color=GREEN)
    ax.axhline(100, color=MUTED, ls="--", lw=1, label="ideal")
    ax.set_xscale("log", base=2)
    ax.set_xticks(cores, [str(c) for c in cores])
    ax.set_xlabel("núcleos alocados (MPI × OpenMP)")
    ax.set_ylabel("eficiência fraca (%)")
    ax.set_title("Híbrido — referência histórica amd-512", loc="left", color=MUTED, fontsize=9)
    ax.legend()
    save(fig, "5_escala_fraca_hibrida.png", "Eficiência da escala fraca híbrida")
    return True


def main():
    rows = target_details()
    if not rows:
        raise SystemExit("Sem medições Target com offload real; gráficos GPU não foram gerados.")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = gpu_summary(rows)
    common = target_common()
    old_seq = seq_historical()
    made = [plot_times(summary, rows, common, old_seq),
            plot_speedup(summary, rows, common, old_seq),
            plot_transfers(summary), plot_hybrid_strong(), plot_hybrid_weak()]
    print(f"{sum(made)} gráfico(s) gravados em {OUT}")
    if not any(made):
        raise SystemExit("Nenhum gráfico pôde ser gerado com os dados disponíveis.")


if __name__ == "__main__":
    main()
