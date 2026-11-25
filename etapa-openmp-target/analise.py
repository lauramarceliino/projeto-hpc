#!/usr/bin/env python3
"""Resume campanhas OpenMP Target e gera resultados/tabelas.md.

Uso: python3 analise.py
Os resultados Target são comparados primeiro com o sequencial do mesmo job.
O sequencial antigo e o híbrido amd-512 aparecem identificados como referências.
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median, stdev

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "resultados"
SEQ_CSV = ROOT.parent / "etapa-sequencial" / "resultados" / "resultados.csv"
HIB_CSV = ROOT.parent / "etapa-hibrida" / "resultados" / "resultados.csv"
HIB_DETAILS = ROOT.parent / "etapa-hibrida" / "resultados" / "detalhes.csv"


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(line for line in f if line.strip() and not line.startswith("#")))


def as_float(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return None


def as_int(row, name, fallback=0):
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError):
        return fallback


def target_details():
    rows = []
    for path in sorted(RESULTS.glob("job_*/detalhes.csv")):
        rows.extend(read_csv(path))
    return [r for r in rows if r.get("versao") == "target" and r.get("offload_real") == "1"]


def target_common():
    rows = []
    for path in sorted(RESULTS.glob("job_*/resultados.csv")):
        rows.extend(read_csv(path))
    return rows


def medians(rows, field, key):
    grouped = defaultdict(list)
    for row in rows:
        value = as_float(row, field)
        if value is not None:
            grouped[key(row)].append(value)
    return {k: median(v) for k, v in grouped.items()}


def seq_same_job(rows):
    return medians(
        [r for r in rows if r.get("versao") == "seq"],
        "tempo_s",
        lambda r: (r.get("job", ""), as_int(r, "N"), as_int(r, "iteracoes")),
    )


def seq_historical():
    return medians(
        [r for r in read_csv(SEQ_CSV) if r.get("versao") == "seq"],
        "tempo_s",
        lambda r: as_int(r, "N"),
    )


def hybrid_reference():
    groups = defaultdict(list)
    for row in read_csv(HIB_CSV):
        if row.get("versao") != "hib":
            continue
        value = as_float(row, "tempo_s")
        if value is None:
            continue
        key = (as_int(row, "N"), as_int(row, "processos"), as_int(row, "threads"))
        groups[key].append(value)
    best = {}
    for (n, p, t), values in groups.items():
        candidate = (median(values), p, t)
        if n not in best or candidate[0] < best[n][0]:
            best[n] = candidate
    return best


def gpu_summary(rows):
    fields = ("tempo_total_s", "tempo_kernel_s", "tempo_entrada_s", "tempo_saida_s",
              "bytes_h2d_estimados", "bytes_d2h_estimados")
    groups = defaultdict(list)
    for row in rows:
        n = as_int(row, "N")
        it = as_int(row, "iteracoes")
        groups[(n, it)].append(row)
    summary = {}
    for key, samples in groups.items():
        values = {field: [as_float(row, field) for row in samples] for field in fields}
        summary[key] = {
            "reps": len(samples),
            "median": {field: median(v for v in items if v is not None)
                       for field, items in values.items() if any(v is not None for v in items)},
            "std": stdev(values["tempo_total_s"]) if len(values["tempo_total_s"]) > 1 else 0.0,
            "devices": sorted({row.get("device", "desconhecido") for row in samples}),
            "nodes": sorted({row.get("no", "desconhecido") for row in samples}),
            "jobs": sorted({row.get("job", "") for row in samples}),
            "partitions": sorted({row.get("particao", "") for row in samples}),
        }
    return summary


def hybrid_weak_table():
    rows = [r for r in read_csv(HIB_DETAILS)
            if r.get("experimento") == "fraca_espalhada" and r.get("versao") == "hib"]
    if not rows:
        return ["Dados de escalabilidade fraca híbrida ainda não encontrados."]
    grouped = defaultdict(list)
    for row in rows:
        key = (as_int(row, "processos") * as_int(row, "threads"), as_int(row, "N"),
               as_int(row, "processos"), as_int(row, "threads"))
        if as_float(row, "tempo_s") is not None:
            grouped[key].append(row)
    base_rows = [r for r in rows if as_int(r, "N") == 4096 and
                 as_int(r, "processos") == 1 and as_int(r, "threads") == 1]
    if not base_rows:
        return ["Não há linha de base híbrida 1x1, N=4096 nos dados fracos."]
    base = median(as_float(r, "tempo_s") for r in base_rows)
    out = ["| núcleos | N | PxT | tempo (s) | eficiência fraca | comunicação |",
           "|---:|---:|---:|---:|---:|---:|"]
    for (cores, n, p, t), samples in sorted(grouped.items()):
        tempo = median(as_float(r, "tempo_s") for r in samples)
        comm_values = [as_float(r, "comm_med_s") for r in samples]
        comm_values = [v for v in comm_values if v is not None]
        comm = median(comm_values) if comm_values else None
        out.append(f"| {cores} | {n} | {p}x{t} | {tempo:.4f} | {100 * base / tempo:.1f}% "
                   f"| {100 * comm / tempo:.1f}% |" if comm is not None else
                   f"| {cores} | {n} | {p}x{t} | {tempo:.4f} | {100 * base / tempo:.1f}% | — |")
    return out


def hybrid_strong_table():
    rows = [r for r in read_csv(HIB_DETAILS)
            if r.get("experimento") == "forte" and r.get("versao") == "hib"
            and as_int(r, "N") == 16384]
    base = seq_historical().get(16384)
    if not rows or not base:
        return ["Dados de escala forte híbrida N=16384 ou baseline sequencial ausentes."]
    groups = defaultdict(list)
    for row in rows:
        key = (as_int(row, "processos"), as_int(row, "threads"))
        if as_float(row, "tempo_s") is not None:
            groups[key].append(row)
    out = ["| núcleos | PxT | tempo (s) | speedup | eficiência | comunicação | desbalanceamento |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for (p, t), samples in sorted(groups.items(), key=lambda item: item[0][0] * item[0][1]):
        tempo = median(as_float(r, "tempo_s") for r in samples)
        speedup = base / tempo
        comm_values = [as_float(r, "comm_med_s") for r in samples]
        imbalance_values = [as_float(r, "desbalanceamento") for r in samples]
        comm_values = [v for v in comm_values if v is not None]
        imbalance_values = [v for v in imbalance_values if v is not None]
        comm = 100 * median(comm_values) / tempo if comm_values else None
        imbalance = median(imbalance_values) if imbalance_values else None
        out.append(f"| {p * t} | {p}x{t} | {tempo:.4f} | {speedup:.2f}x | "
                   f"{100 * speedup / (p * t):.1f}% | "
                   f"{comm:.1f}% | {imbalance:.3f} |" if comm is not None and imbalance is not None else
                   f"| {p * t} | {p}x{t} | {tempo:.4f} | {speedup:.2f}x | "
                   f"{100 * speedup / (p * t):.1f}% | — | — |")
    return out


def main():
    raw_target = target_details()
    if not raw_target:
        print("Nenhuma medição OpenMP Target com offload_real=1 encontrada.")
        print("Submeta run_target.slurm no NPAD antes de gerar a comparação de desempenho.")
        return 0

    summary = gpu_summary(raw_target)
    common = target_common()
    local_seq = seq_same_job(common)
    old_seq = seq_historical()
    hybrid = hybrid_reference()

    lines = [
        "# Resultados — OpenMP Target",
        "",
        "Os tempos Target abaixo só incluem observações com `offload_real=1`. "
        "O speedup usa o sequencial do mesmo job quando disponível; caso contrário, "
        "a coluna identifica o baseline histórico `amd-512`.",
        "",
        "## GPU e referência sequencial",
        "",
        "| N | iterações | reps | seq mediana (s) | origem seq | Target total mediana (s) | dp (s) | "
        "kernel (s) | entrada (s) | saída (s) | speedup | híbrido ref. (s; PxT) |",
        "|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for (n, it), item in sorted(summary.items()):
        matched_seq = [local_seq[(job, n, it)] for job in item["jobs"]
                       if (job, n, it) in local_seq]
        seq_value = median(matched_seq) if matched_seq else None
        seq_source = "mesmo job/nó" if seq_value is not None else "amd-512 (histórico)"
        if seq_value is None:
            seq_value = old_seq.get(n)
        m = item["median"]
        speedup = seq_value / m["tempo_total_s"] if seq_value and m.get("tempo_total_s", 0) else None
        speedup_text = f"{speedup:.2f}x" if speedup is not None else "—"
        hybrid_cell = "—"
        if n in hybrid:
            htime, p, t = hybrid[n]
            hybrid_cell = f"{htime:.4f}; {p}x{t} (amd-512)"
        lines.append(
            f"| {n} | {it} | {item['reps']} | {seq_value:.6f} | {seq_source} | "
            f"{m['tempo_total_s']:.6f} | {item['std']:.6f} | {m['tempo_kernel_s']:.6f} | "
            f"{m['tempo_entrada_s']:.6f} | {m['tempo_saida_s']:.6f} | {speedup_text} | {hybrid_cell} |"
            if seq_value is not None else
            f"| {n} | {it} | {item['reps']} | — | — | {m['tempo_total_s']:.6f} | "
            f"{item['std']:.6f} | {m['tempo_kernel_s']:.6f} | {m['tempo_entrada_s']:.6f} | "
            f"{m['tempo_saida_s']:.6f} | — | {hybrid_cell} |"
        )

    lines.extend([
        "",
        "`tempo_total` inclui mapeamento, kernels, verificações opcionais e leitura final. "
        "Entrada e saída são tempos agregados do runtime, não medição pura de PCIe. "
        "O tempo híbrido é referência histórica em hardware AMD e não deve ser tratado como "
        "comparação controlada com a GPU.",
        "",
        "## Transferências estimadas",
        "",
        "| N | bytes H2D | bytes D2H | dispositivo(s) | partição/job |",
        "|---:|---:|---:|:---|:---|",
    ])
    for (n, it), item in sorted(summary.items()):
        m = item["median"]
        lines.append(f"| {n} | {m['bytes_h2d_estimados']:.0f} | {m['bytes_d2h_estimados']:.0f} | "
                     f"{', '.join(item['devices'])} | {', '.join(item['partitions'])}/{', '.join(item['jobs'])} |")

    lines.extend([
        "",
        "## Escalabilidade forte e fraca",
        "",
        "A tabela de tempo e speedup mantém o mesmo tamanho de problema por linha; "
        "a versão OpenMP Target usa uma GPU fixa. As variações de tamanho não são chamadas "
        "de escalabilidade fraca da GPU. Eficiência paralela `S/P` aplica-se às configurações "
        "CPU/MPI+OpenMP com `P` núcleos; a GPU é descrita por speedup, atualizações/s e "
        "custos de dados.",
        "",
        "### Híbrido: escala forte (referência histórica amd-512, N=16384)",
        "",
    ])
    lines.extend(hybrid_strong_table())
    lines.extend([
        "",
        "A tabela híbrida abaixo reproduz a escalabilidade fraca com afinidade espalhada registrada "
        "na etapa MPI+OpenMP. Ela é mantida separada por usar resultados e hardware próprios.",
        "",
    ])
    lines.extend(hybrid_weak_table())
    lines.extend([
        "",
        "## Comunicação MPI e balanceamento",
        "",
        "A comunicação e o desbalanceamento são métricas da implementação híbrida; "
        "as transferências H2D/D2H são métricas distintas da versão GPU. Consulte "
        "`../etapa-hibrida/analise.py` e `../etapa-hibrida/resultados/tabelas.md` para a análise "
        "completa dessas medidas.",
        "",
    ])
    destination = RESULTS / "tabelas.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(lines))
    print(f"Tabelas gravadas em {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
