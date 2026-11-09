#!/usr/bin/env python3
"""Gráficos da versão híbrida para os slides (PNG em graficos/).
Uso: python3 graficos.py   (requer matplotlib)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analise import load, seq_baseline

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, AXIS, BG = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
CORES = [1, 2, 4, 8, 16, 32, 64, 128]
OUT_DIR = "graficos"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
    "legend.frameon": False,
})


def log2_axes(ax, ylabel, ideal=True):
    """Eixo x em núcleos (log2); com ideal=True, eixo y também log2 e a reta de speedup ideal."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(CORES, [str(n) for n in CORES])
    ax.set_xlabel("núcleos")
    ax.set_ylabel(ylabel)
    if ideal:
        ax.set_yscale("log", base=2)
        ax.plot(CORES, CORES, ls="--", lw=1.2, color=MUTED, label="ideal")


def plot_line(ax, x, y, i, label, at=-1, offset=(8, 0)):
    """Linha da série i; com at != None, rótulo direto no ponto de índice 'at'."""
    ax.plot(x, y, lw=2, color=COLORS[i], marker="o", ms=6, mec=BG, mew=1.5, label=label)
    if at is not None:
        ax.annotate(label, (x[at], y[at]), xytext=offset, textcoords="offset points",
                    ha="left" if offset[0] > 0 else "center", va="center", color=INK, fontsize=10)


def save(fig, name, title):
    fig.suptitle(title, x=0.06, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=200)
    plt.close(fig)


def time_1xT(data, compact, n, c):
    """Tempo de 1 processo x c threads (1x1 vem sempre da afinidade compacta)."""
    return (compact if c == 1 else data)[("hib", n, 1, c)]["tempo_s"]


def strong_speedup(seq, compact, spread):
    """Speedup x núcleos para três tamanhos (afinidade espalhada)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    log2_axes(ax, "speedup sobre o sequencial")
    for i, n in enumerate((4096, 8192, 16384)):
        plot_line(ax, CORES, [seq[n] / time_1xT(spread, compact, n, c) for c in CORES], i, f"N={n}")
    ax.set_yticks(CORES, [str(n) for n in CORES])
    ax.legend(loc="upper left")
    save(fig, "1_speedup_forte.png", "Escala forte (1 processo × T threads, afinidade espalhada)")


def affinity(seq, compact, spread):
    """Afinidade compacta x espalhada para N=16384."""
    n = 16384
    fig, ax = plt.subplots(figsize=(9, 5))
    log2_axes(ax, "speedup sobre o sequencial")
    for i, (name, data) in enumerate((("espalhada", spread), ("compacta", compact))):
        y = [seq[n] / time_1xT(data, compact, n, c) for c in CORES]
        plot_line(ax, CORES, y, i, name, at=3, offset=(0, 14 if i == 0 else -14))
    ax.set_yticks(CORES, [str(n) for n in CORES])
    ax.axhline(10, color=AXIS, lw=1)
    ax.text(1, 10.6, "limite de banda de memória do nó (~10×)", color=INK2, fontsize=9)
    ax.legend(loc="upper left")
    save(fig, "2_afinidade.png", f"Afinidade: compacta × espalhada (N={n}, 1 processo × T threads)")


def weak_efficiency(weak):
    """Eficiência da escala fraca para as séries MPI, OpenMP e híbrida."""
    base = weak[("hib", 4096, 1, 1)]["tempo_s"]
    series = {"MPI (P×1)": lambda c: (c, 1),
              "OpenMP (1×T)": lambda c: (1, c),
              "Híbrido": lambda c: {1: (1, 1), 4: (2, 2), 16: (8, 2), 32: (8, 4),
                                    64: (8, 8), 128: (8, 16)}.get(c)}
    fig, ax = plt.subplots(figsize=(9, 5))
    log2_axes(ax, "eficiência (%)", ideal=False)
    for i, (name, config) in enumerate(series.items()):
        points = []
        for c in CORES:
            pt = config(c)
            keys = [k for k in weak if pt and k[0] == "hib" and k[2:] == pt]
            if keys:
                points.append((c, 100 * base / weak[keys[0]]["tempo_s"]))
        plot_line(ax, [p[0] for p in points], [p[1] for p in points], i, name, at=None)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower left")
    save(fig, "3_escala_fraca.png", "Escala fraca (N = 4096·√núcleos, afinidade espalhada)")


def communication(compact):
    """Fração do laço gasta em comunicação para cada divisão PxT em 128 núcleos."""
    n = 16384
    configs = [(1, 128), (2, 64), (4, 32), (8, 16), (16, 8), (32, 4), (64, 2), (128, 1)]
    values = [100 * compact[("hib", n, p, t)]["comm_med_s"] / compact[("hib", n, p, t)]["tempo_s"]
              for p, t in configs]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_axisbelow(True)
    ax.bar([f"{p}×{t}" for p, t in configs], values, width=0.6, color=COLORS[0], edgecolor=BG, linewidth=2)
    for x, v in enumerate(values):
        ax.text(x, v + 0.4, f"{v:.1f}%".replace(".", ","), ha="center", color=INK, fontsize=10)
    ax.set_xlabel("processos MPI × threads OpenMP (128 núcleos)")
    ax.set_ylabel("tempo em comunicação (% do laço)")
    ax.grid(axis="x", visible=False)
    ax.set_yticks(range(0, 25, 5))
    save(fig, "4_comunicacao.png", f"Overhead de comunicação MPI (N={n}, mesmo tempo total ≈ 2,05 s)")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    seq, compact, spread = seq_baseline(), load("forte"), load("forte_espalhada")
    strong_speedup(seq, compact, spread)
    affinity(seq, compact, spread)
    weak_efficiency(load("fraca_espalhada"))
    communication(compact)
