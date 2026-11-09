/*
 * heat_hib.c — difusão de calor 2D, MPI + OpenMP.
 * MPI: blocos de linhas com halo de 1 linha. OpenMP: linhas do bloco entre threads.
 * Resultado idêntico bit a bit ao heat_seq.
 * Uso: mpirun -np P ./heat_hib -n N -i ITER [-c CASO] [-t TOL] [-o ARQ] [-a] [-b]
 */
#include "heat_common.h"

#include <mpi.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>

#define TAG_DOWN 1
#define TAG_UP   2

static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int imax(int a, int b) { return a > b ? a : b; }

/* Bloco de linhas de um rank. Linhas locais 0 e nloc+1 são o halo;
 * a linha local l é a global first + l - 1; lo..hi são as linhas a atualizar. */
typedef struct {
    int n;
    int rank, nprocs;
    int first;
    int nloc;
    int up, down;
    int lo, hi;
} Domain;

/* Divide as n linhas em blocos (diferença máxima de 1 linha entre ranks). */
static Domain domain_create(int n, int rank, int nprocs)
{
    Domain d;
    int base = n / nprocs, rest = n % nprocs;
    d.n = n;
    d.rank = rank;
    d.nprocs = nprocs;
    d.nloc = base + (rank < rest);
    d.first = rank * base + imin(rank, rest);
    d.up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    d.down = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;
    d.lo = imax(1, 2 - d.first);
    d.hi = imin(d.nloc, n - 1 - d.first);
    return d;
}

static inline int owns_row(const Domain *d, int g) { return g >= d->first && g < d->first + d->nloc; }
static inline int local_row(const Domain *d, int g) { return g - d->first + 1; }

/* Aloca o bloco local com as 2 linhas de halo. */
static double *alloc_local(const Domain *d)
{
    double *p = (double *)malloc((size_t)(d->nloc + 2) * (size_t)d->n * sizeof(double));
    if (!p) {
        fprintf(stderr, "[rank %d] Erro: sem memória para %d x %d linhas locais\n", d->rank,
                d->nloc + 2, d->n);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return p;
}

/* Inicia a troca de halo: envia a primeira/última linha própria e recebe as vizinhas. */
static void halo_start(double *u, const Domain *d, MPI_Request req[4])
{
    int n = d->n;
    MPI_Irecv(&u[IDX(0, 0, n)], n, MPI_DOUBLE, d->up, TAG_DOWN, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(&u[IDX(d->nloc + 1, 0, n)], n, MPI_DOUBLE, d->down, TAG_UP, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(&u[IDX(1, 0, n)], n, MPI_DOUBLE, d->up, TAG_UP, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(&u[IDX(d->nloc, 0, n)], n, MPI_DOUBLE, d->down, TAG_DOWN, MPI_COMM_WORLD, &req[3]);
}

/* Troca de halo bloqueante. */
static void halo_exchange(double *u, const Domain *d)
{
    MPI_Request req[4];
    halo_start(u, d, req);
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
}

/* Stencil de 5 pontos nas linhas locais a..b, divididas entre as threads.
 * Mesma ordem de soma do heat_seq: resultado idêntico bit a bit. */
static void update_rows(const double *restrict u, double *restrict un, int a, int b, int n)
{
#pragma omp parallel for schedule(static)
    for (int l = a; l <= b; l++)
        for (int j = 1; j < n - 1; j++)
            un[IDX(l, j, n)] = 0.25 * (u[IDX(l - 1, j, n)] + u[IDX(l + 1, j, n)] +
                                       u[IDX(l, j - 1, n)] + u[IDX(l, j + 1, n)]);
}

/* Stencil só nas linhas que leem o halo (primeira e última); divide as colunas entre as threads. */
static void update_edges(const double *restrict u, double *restrict un, const Domain *d)
{
    int rows[2], k = 0, n = d->n;
    if (d->lo <= 1 && 1 <= d->hi) rows[k++] = 1;
    if (d->nloc > 1 && d->lo <= d->nloc && d->nloc <= d->hi) rows[k++] = d->nloc;
    if (k == 0) return;

#pragma omp parallel for collapse(2) schedule(static)
    for (int q = 0; q < k; q++)
        for (int j = 1; j < n - 1; j++) {
            int l = rows[q];
            un[IDX(l, j, n)] = 0.25 * (u[IDX(l - 1, j, n)] + u[IDX(l + 1, j, n)] +
                                       u[IDX(l, j - 1, n)] + u[IDX(l, j + 1, n)]);
        }
}

/* Um passo de Jacobi. Com overlap, calcula o miolo enquanto o halo trafega.
 * Acumula em t_comp/t_comm o tempo de cálculo e de comunicação deste rank. */
static void heat_step_hib(double *u, double *un, const Domain *d, int overlap,
                          double *t_comp, double *t_comm)
{
    double a = MPI_Wtime(), b, c;
    if (overlap) {
        MPI_Request req[4];
        halo_start(u, d, req);
        b = MPI_Wtime();
        update_rows(u, un, imax(d->lo, 2), imin(d->hi, d->nloc - 1), d->n);
        c = MPI_Wtime();
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        double e = MPI_Wtime();
        update_edges(u, un, d);
        double f = MPI_Wtime();
        *t_comm += (b - a) + (e - c);
        *t_comp += (c - b) + (f - e);
    } else {
        halo_exchange(u, d);
        b = MPI_Wtime();
        update_rows(u, un, d->lo, d->hi, d->n);
        c = MPI_Wtime();
        *t_comm += b - a;
        *t_comp += c - b;
    }
}

/* Gera as linhas locais (e o halo) nos dois buffers. Paralelo para o "first touch"
 * colocar cada página no nó NUMA da thread que vai usá-la. */
static void init_local(double *u, double *un, const Domain *d, int test_case)
{
    int n = d->n;
#pragma omp parallel for schedule(static)
    for (int l = 0; l <= d->nloc + 1; l++) {
        int g = d->first + l - 1;
        for (int j = 0; j < n; j++) {
            double v = (g >= 0 && g < n) ? heat_valor_inicial(test_case, g, j, n) : 0.0;
            u[IDX(l, j, n)] = v;
            un[IDX(l, j, n)] = v;
        }
    }
}

/* Resíduo global: max |média dos vizinhos - u| (igual ao heat_residuo). */
static double global_residual(double *u, const Domain *d)
{
    int n = d->n;
    double r = 0.0, rg;
    halo_exchange(u, d);
#pragma omp parallel for schedule(static) reduction(max : r)
    for (int l = d->lo; l <= d->hi; l++)
        for (int j = 1; j < n - 1; j++) {
            double mean = 0.25 * (u[IDX(l - 1, j, n)] + u[IDX(l + 1, j, n)] +
                                  u[IDX(l, j - 1, n)] + u[IDX(l, j + 1, n)]);
            double diff = fabs(mean - u[IDX(l, j, n)]);
            if (diff > r) r = diff;
        }
    MPI_Allreduce(&r, &rg, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return rg;
}

/* Reúne no rank 0, em ordem global, 'per_row' valores por linha de cada rank. */
static void gather_rows(const double *local, double *global, int per_row, const Domain *d)
{
    int *counts = NULL, *displs = NULL;
    if (d->rank == 0) {
        counts = (int *)malloc(d->nprocs * sizeof(int));
        displs = (int *)malloc(d->nprocs * sizeof(int));
        for (int r = 0; r < d->nprocs; r++) {
            Domain dr = domain_create(d->n, r, d->nprocs);
            counts[r] = dr.nloc * per_row;
            displs[r] = dr.first * per_row;
        }
    }
    MPI_Gatherv(local, d->nloc * per_row, MPI_DOUBLE, global, counts, displs, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    free(counts);
    free(displs);
}

/* Assinatura global, idêntica à do sequencial: somas por linha juntadas no rank 0
 * e acumuladas na mesma ordem; o resultado é difundido para todos os ranks. */
static HeatStats global_stats(const double *u, const Domain *d)
{
    int n = d->n;
    double *row_sums = (double *)malloc((size_t)d->nloc * 2 * sizeof(double));
    double mn = u[IDX(1, 0, n)], mx = mn;

#pragma omp parallel for schedule(static) reduction(min : mn) reduction(max : mx)
    for (int l = 1; l <= d->nloc; l++) {
        double s = 0.0, s2 = 0.0;
        for (int j = 0; j < n; j++) {
            double v = u[IDX(l, j, n)];
            s += v;
            s2 += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        row_sums[2 * (l - 1)] = s;
        row_sums[2 * (l - 1) + 1] = s2;
    }

    double *all = (d->rank == 0) ? (double *)malloc((size_t)n * 2 * sizeof(double)) : NULL;
    gather_rows(row_sums, all, 2, d);
    free(row_sums);

    /* 4 células do centro e u[1][n/2]: quem não é dono manda 0 (x + 0 = x exato) */
    int m = n / 2;
    int gr[5] = {m - 1, m - 1, m, m, 1};
    int gc[5] = {m - 1, m, m - 1, m, m};
    double pt[5] = {0, 0, 0, 0, 0}, ptg[5];
    for (int k = 0; k < 5; k++)
        if (owns_row(d, gr[k])) pt[k] = u[IDX(local_row(d, gr[k]), gc[k], n)];

    double mng, mxg;
    MPI_Reduce(&mn, &mng, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mx, &mxg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(pt, ptg, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    HeatStats s;
    if (d->rank == 0) {
        double sum = 0.0, sum2 = 0.0;
        for (int i = 0; i < n; i++) {
            sum += all[2 * i];
            sum2 += all[2 * i + 1];
        }
        free(all);
        s.soma = sum;
        s.norma_l2 = sqrt(sum2);
        s.minimo = mng;
        s.maximo = mxg;
        s.centro = (n % 2) ? ptg[3] : 0.25 * (ptg[0] + ptg[1] + ptg[2] + ptg[3]);
        s.perto_borda = ptg[4];
    }
    MPI_Bcast(&s, sizeof(HeatStats), MPI_BYTE, 0, MPI_COMM_WORLD);
    return s;
}

/* Grava a grade global em binário (mesmo formato do heat_seq -o) com MPI-IO coletivo. */
static int write_grid(const char *path, const double *u, const Domain *d)
{
    MPI_File fh;
    MPI_Datatype row;
    if (MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                      &fh) != MPI_SUCCESS)
        return 0;
    MPI_File_set_size(fh, 0);
    MPI_Type_contiguous(d->n, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    MPI_Offset off = (MPI_Offset)d->first * d->n * (MPI_Offset)sizeof(double);
    int rc = MPI_File_write_at_all(fh, off, &u[IDX(1, 0, d->n)], d->nloc, row, MPI_STATUS_IGNORE);
    MPI_Type_free(&row);
    MPI_File_close(&fh);
    return rc == MPI_SUCCESS;
}

/* Mesmos critérios de validação do heat_seq. Devolve 1 se tudo passou. */
static int validate(const double *u, const Domain *d, int test_case, const HeatStats *s,
                    int analytic, double tol, double residual)
{
    int n = d->n, ok = 1, root = (d->rank == 0);
    if (root) printf("-- validação --\n");

    int maxp = (s->minimo >= T_FRIO && s->maximo <= T_QUENTE);
    if (root) printf("princípio do máximo (0 <= T <= 100)  : %s\n", maxp ? "OK" : "FALHOU");
    ok &= maxp;

    if (test_case == CASO_UNIFORME || test_case == CASO_LINEAR) {
        double err = 0.0, err_g;
#pragma omp parallel for schedule(static) reduction(max : err)
        for (int l = 1; l <= d->nloc; l++) {
            int g = d->first + l - 1;
            for (int j = 0; j < n; j++) {
                double e = fabs(u[IDX(l, j, n)] - heat_valor_inicial(test_case, g, j, n));
                if (e > err) err = e;
            }
        }
        MPI_Allreduce(&err, &err_g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        int r = err_g < 1e-10;
        if (root) printf("ponto fixo: erro máx. = %.3e          : %s\n", err_g, r ? "OK" : "FALHOU");
        ok &= r;
    }

    if (analytic) {
        if (test_case != CASO_PLACA || n > 1025) {
            if (root) printf("-a ignorado (só vale para o caso 0 com N <= 1025)\n");
        } else {
            int conv = (tol > 0.0 && residual < tol);
            if (root) printf("convergiu (resíduo < tol)            : %s\n", conv ? "OK" : "NÃO (aumente -i)");

            double center_err = fabs(s->centro - 0.25 * T_QUENTE);
            int rc = center_err < 1e-4;
            if (root)
                printf("centro = %.8f (esperado 25)  erro = %.2e : %s\n", s->centro, center_err,
                       rc ? "OK" : "FALHOU");

            double sq = 0.0, emax = 0.0, sq_g, emax_g;
#pragma omp parallel for schedule(static) reduction(+ : sq) reduction(max : emax)
            for (int l = d->lo; l <= d->hi; l++) {
                int g = d->first + l - 1;
                for (int j = 1; j < n - 1; j++) {
                    double e = fabs(u[IDX(l, j, n)] - heat_analitica(g, j, n));
                    sq += e * e;
                    if (e > emax) emax = e;
                }
            }
            MPI_Allreduce(&sq, &sq_g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&emax, &emax_g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            double rms = sqrt(sq_g / ((double)(n - 2) * (double)(n - 2)));
            int ra = rms < 0.5;
            if (root)
                printf("vs analítica: RMS = %.3e  máx = %.3e (°C) : %s\n", rms, emax_g,
                       ra ? "OK" : "FALHOU");
            ok &= conv && rc && ra;
        }
    }

    if (root) printf("VALIDACAO: %s\n", ok ? "OK" : "FALHOU");
    return ok;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Uso: mpirun -np P %s [-n N] [-i ITER] [-c CASO] [-t TOL] [-o ARQ] [-a] [-b]\n"
            "  ou mpirun -np P %s N ITER    (modo PaScal: enxuto)\n"
            "  -n N     lado da grade (N x N)                          [1024]\n"
            "  -i ITER  número (máximo) de iterações                   [1000]\n"
            "  -c CASO  0=placa 1=uniforme 2=linear 3=ruído            [0]\n"
            "  -t TOL   para antes se o resíduo < TOL (só validação)   [desligado]\n"
            "  -o ARQ   grava a grade final em binário (use N pequeno)\n"
            "  -a       compara com a solução analítica (caso 0; use com -t e N <= 1025)\n"
            "  -b       troca de halo bloqueante (sem sobrepor comunicação e cálculo)\n"
            "  Threads por processo: variável OMP_NUM_THREADS\n",
            prog, prog);
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int root = (rank == 0);

    int n = 1024, iters = 1000, test_case = CASO_PLACA, analytic = 0, overlap = 1, opt;
    double tol = 0.0;
    const char *out_path = NULL;

    while ((opt = getopt(argc, argv, "n:i:c:t:o:abh")) != -1) {
        switch (opt) {
        case 'n': n = atoi(optarg); break;
        case 'i': iters = atoi(optarg); break;
        case 'c': test_case = atoi(optarg); break;
        case 't': tol = atof(optarg); break;
        case 'o': out_path = optarg; break;
        case 'a': analytic = 1; break;
        case 'b': overlap = 0; break;
        default:
            if (root) usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    /* Modo PaScal: argumentos posicionais N ITER, saída enxuta (como no heat_seq) */
    int pascal_mode = (optind < argc);
    if (pascal_mode) {
        n = atoi(argv[optind]);
        if (optind + 1 < argc) iters = atoi(argv[optind + 1]);
    }
    if (n < 3 || iters < 0 || test_case < 0 || test_case > 3 || n < nprocs) {
        if (root) {
            if (n >= 3 && n < nprocs)
                fprintf(stderr, "Erro: N (%d) menor que o número de processos (%d)\n", n, nprocs);
            usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED && root)
        fprintf(stderr, "Aviso: a biblioteca MPI não garante MPI_THREAD_FUNNELED\n");

    int nthreads = omp_get_max_threads();
    Domain d = domain_create(n, rank, nprocs);

    double *u = alloc_local(&d);
    double *un = alloc_local(&d);
    init_local(u, un, &d, test_case);

    if (!pascal_mode && root) {
        Domain last = domain_create(n, nprocs - 1, nprocs);
        printf("== heat_hib (MPI + OpenMP) ==\n");
        printf("N               = %d (%.2f milhões de pontos, %.1f MB por buffer global)\n", n,
               (double)n * n / 1e6, (double)n * n * sizeof(double) / 1e6);
        printf("caso            = %d\n", test_case);
        printf("processos       = %d\n", nprocs);
        printf("threads/proc    = %d\n", nthreads);
        printf("linhas/proc     = %d a %d\n", last.nloc, d.nloc);
        printf("halo            = %s\n", overlap ? "não bloqueante, sobreposto ao cálculo"
                                                 : "bloqueante");
    }

    /* Laço temporal: única parte cronometrada */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_comp = 0.0, t_comm = 0.0;
    double t0 = MPI_Wtime();
    int it = 0;
    for (; it < iters; it++) {
        heat_step_hib(u, un, &d, overlap, &t_comp, &t_comm);
        double *tmp = u; u = un; un = tmp;

        if (tol > 0.0 && (it + 1) % 100 == 0 && global_residual(u, &d) < tol) {
            it++;
            break;
        }
    }
    double t_local = MPI_Wtime() - t0, elapsed;
    MPI_Allreduce(&t_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double times[2] = {t_comp, t_comm}, tmax[2], tsum[2];
    MPI_Reduce(times, tmax, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(times, tsum, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (pascal_mode) {
        double *diag = (double *)malloc((size_t)d.nloc * sizeof(double));
        for (int l = 1; l <= d.nloc; l++) diag[l - 1] = u[IDX(l, d.first + l - 1, n)];
        double *all = root ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
        gather_rows(diag, all, 1, &d);
        if (root) {
            double trace = 0.0;
            for (int i = 0; i < n; i++) trace += all[i];
            printf("N: %d | Iter: %d | Procs: %d | Threads: %d | Tempo: %.6f s | Trace: %.15e\n",
                   n, it, nprocs, nthreads, elapsed, trace);
        }
        free(diag);
        free(all);
        free(u);
        free(un);
        MPI_Finalize();
        return 0;
    }

    HeatStats s = global_stats(u, &d);
    double residual = global_residual(u, &d);

    if (root) {
        const char *version = overlap ? "hib" : "hib_bloq";
        double mpoints = (double)(n - 2) * (n - 2) * it / (elapsed > 0 ? elapsed : 1e-9) / 1e6;
        printf("iteracoes       = %d\n", it);
        printf("tempo_loop_s    = %.6f\n", elapsed);
        printf("mpontos_por_s   = %.1f\n", mpoints);
        printf("soma            = %.15e\n", s.soma);
        printf("norma_L2        = %.15e\n", s.norma_l2);
        printf("minimo          = %.15e\n", s.minimo);
        printf("maximo          = %.15e\n", s.maximo);
        printf("centro          = %.15e\n", s.centro);
        printf("perto_da_borda  = %.15e\n", s.perto_borda);
        printf("residuo         = %.6e\n", residual);
        printf("CSV,%s,%d,%d,%d,%d,%.6f\n", version, n, it, nprocs, nthreads, elapsed);

        double comp_avg = tsum[0] / nprocs, comm_avg = tsum[1] / nprocs;
        double imbalance = comp_avg > 0 ? tmax[0] / comp_avg : 1.0;
        printf("calculo_s       = max %.6f | media %.6f\n", tmax[0], comp_avg);
        printf("comunicacao_s   = max %.6f | media %.6f (%.1f%% do laço)\n", tmax[1], comm_avg,
               100.0 * comm_avg / (elapsed > 0 ? elapsed : 1e-9));
        printf("desbalanceamento= %.4f (calculo max / medio)\n", imbalance);
        printf("DET,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f\n", version, n, it, nprocs,
               nthreads, elapsed, tmax[0], comp_avg, tmax[1], comm_avg, imbalance);
    }

    int io_error = 0;
    if (out_path) {
        int io_ok = write_grid(out_path, u, &d), all_ok;
        MPI_Allreduce(&io_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!all_ok) {
            if (root) fprintf(stderr, "Erro ao gravar %s\n", out_path);
            io_error = 1;
        } else if (root) {
            printf("grade gravada em %s\n", out_path);
        }
    }

    int ok = validate(u, &d, test_case, &s, analytic, tol, residual);

    free(u);
    free(un);
    MPI_Finalize();
    return io_error ? 1 : (ok ? 0 : 2);
}
