#!/usr/bin/env python3
"""Tabelas de desempenho da versão híbrida (mediana das repetições).
Uso: python3 analise.py > resultados/tabelas.md
"""
import csv
from collections import defaultdict
from statistics import median

DETAILS = "resultados/detalhes.csv"
SEQ = "../etapa-sequencial/resultados/resultados.csv"
FIELDS = ("tempo_s", "calc_max_s", "calc_med_s", "comm_max_s", "comm_med_s", "desbalanceamento")


def load(experiment):
    """Mediana de cada métrica por (versao, N, processos, threads) de um experimento."""
    groups = defaultdict(list)
    with open(DETAILS) as f:
        for r in csv.DictReader(f):
            if r["experimento"] == experiment:
                groups[(r["versao"], int(r["N"]), int(r["processos"]), int(r["threads"]))].append(r)
    return {k: {c: median(float(r[c]) for r in rows) for c in FIELDS} for k, rows in groups.items()}


def seq_baseline():
    """Mediana do tempo sequencial por N."""
    times = defaultdict(list)
    with open(SEQ) as f:
        for r in csv.DictReader(f):
            times[int(r["N"])].append(float(r["tempo_s"]))
    return {n: median(v) for n, v in times.items()}


def strong(experiment):
    """Escala forte: speedup e eficiência sobre o sequencial, comunicação e desbalanceamento."""
    data, seq = load(experiment), seq_baseline()
    for version in ("hib", "hib_bloq"):
        rows = [(k, m) for k, m in data.items() if k[0] == version and k[1] in seq]
        if not rows:
            continue
        print(f"\n## {experiment} — {version} (speedup e eficiência vs sequencial)\n")
        print("| N | PxT | núcleos | tempo (s) | speedup | eficiência | comunicação | desbal. |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for (v, n, p, t), m in sorted(rows, key=lambda x: (x[0][1], x[0][2] * x[0][3], x[0][2])):
            s = seq[n] / m["tempo_s"]
            print(f"| {n} | {p}x{t} | {p * t} | {m['tempo_s']:.4f} | {s:.2f} | {100 * s / (p * t):.0f}% "
                  f"| {100 * m['comm_med_s'] / m['tempo_s']:.1f}% | {m['desbalanceamento']:.3f} |")


def weak(experiment):
    """Escala fraca: eficiência = T(1 núcleo, N=4096) / T, por série (MPI, OpenMP, híbrida)."""
    data = load(experiment)
    series = {"MPI (Px1)": lambda p, t: t == 1,
              "OpenMP (1xT)": lambda p, t: p == 1,
              "Híbrido (P>1 e T>1)": lambda p, t: (p > 1 and t > 1) or p * t == 1}
    for version in ("hib", "hib_bloq"):
        base = data.get((version, 4096, 1, 1))
        if not base:
            continue
        for name, keep in series.items():
            print(f"\n## {experiment} — {version} — {name} (eficiência = T(1 núcleo, N=4096) / T)\n")
            print("| núcleos | N | PxT | tempo (s) | eficiência | comunicação |")
            print("|---:|---:|---:|---:|---:|---:|")
            for (v, n, p, t), m in sorted(data.items(), key=lambda x: x[0][2] * x[0][3]):
                if v == version and keep(p, t):
                    print(f"| {p * t} | {n} | {p}x{t} | {m['tempo_s']:.4f} "
                          f"| {100 * base['tempo_s'] / m['tempo_s']:.0f}% "
                          f"| {100 * m['comm_med_s'] / m['tempo_s']:.1f}% |")


def affinity():
    """Speedup com afinidade compacta x espalhada, nas mesmas configurações."""
    comp, spread, seq = load("forte"), load("forte_espalhada"), seq_baseline()
    print("\n## Afinidade compacta x espalhada (speedup vs sequencial)\n")
    print("| N | PxT | núcleos | compacta | espalhada |")
    print("|---:|---:|---:|---:|---:|")
    for k in sorted(spread, key=lambda k: (k[1], k[2] * k[3], k[2])):
        v, n, p, t = k
        if v == "hib" and n in seq:
            c = f"{seq[n] / comp[k]['tempo_s']:.2f}" if k in comp else "—"
            print(f"| {n} | {p}x{t} | {p * t} | {c} | {seq[n] / spread[k]['tempo_s']:.2f} |")


if __name__ == "__main__":
    strong("forte")
    weak("fraca")
    strong("forte_espalhada")
    weak("fraca_espalhada")
    affinity()
