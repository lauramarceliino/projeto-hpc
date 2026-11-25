/*
 * heat_target.c — difusão de calor 2D com offloading OpenMP Target.
 *
 * Os dois buffers ficam mapeados durante toda a evolução. Cada região target
 * atualiza somente o interior; a paridade escolhe a origem e o destino Jacobi.
 * A medição separa mapeamento inicial, kernels, verificações opcionais e leitura.
 *
 * Uso: ./build/heat_target -n N -i ITER [-c CASO] [-t TOL] [-o ARQ] [-a] [-H]
 * -H seleciona um caminho host-only para testes locais; campanhas NPAD não devem
 * usar essa opção, pois ela não executa os kernels target.
 */
#include "../etapa-sequencial/heat_common.h"

#include <omp.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    double total_s, entrada_s, kernel_s, verificacao_s, saida_s;
    int iteracoes, verificacoes, kernel_offload;
} HeatTiming;

/* Fallback de teste local; não representa execução OpenMP Target na GPU. */
static void heat_passo_host(double *restrict a, double *restrict b, int n, int iter)
{
    if ((iter & 1) == 0) {
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                b[IDX(i, j, n)] =
                    0.25 * (a[IDX(i - 1, j, n)] + a[IDX(i + 1, j, n)] +
                            a[IDX(i, j - 1, n)] + a[IDX(i, j + 1, n)]);
    } else {
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                a[IDX(i, j, n)] =
                    0.25 * (b[IDX(i - 1, j, n)] + b[IDX(i + 1, j, n)] +
                            b[IDX(i, j - 1, n)] + b[IDX(i, j + 1, n)]);
    }
}

static HeatTiming heat_evolui_host(double *a, double *b, int n, int max_iter, double tol)
{
    HeatTiming t = {0};
    double start = heat_wtime();
    for (int it = 0; it < max_iter; it++) {
        double t0 = heat_wtime();
        heat_passo_host(a, b, n, it);
        t.kernel_s += heat_wtime() - t0;
        t.iteracoes = it + 1;
        if (tol > 0.0 && t.iteracoes % 100 == 0) {
            double tv = heat_wtime();
            const double *atual = (t.iteracoes & 1) ? b : a;
            double residuo = heat_residuo(atual, n);
            t.verificacao_s += heat_wtime() - tv;
            t.verificacoes++;
            if (residuo < tol) break;
        }
    }
    t.total_s = heat_wtime() - start;
    return t;
}

static HeatTiming heat_evolui_device(double *a, double *b, int *flags, size_t flag_count,
                                     int n, size_t cells, int max_iter, double tol)
{
    HeatTiming t = {0};
    double start = heat_wtime();

/* Os mapas to mantêm grades e sentinelas residentes; flags confirmam cada kernel. */
#pragma omp target data map(to : a[0:cells], b[0:cells], flags[0:flag_count])
    {
        t.entrada_s = heat_wtime() - start;
        for (int it = 0; it < max_iter; it++) {
            double t0 = heat_wtime();
            if ((it & 1) == 0) {
#pragma omp target teams distribute parallel for collapse(2) map(alloc : a[0:cells], b[0:cells], flags[0:flag_count])
                for (int i = 1; i < n - 1; i++)
                    for (int j = 1; j < n - 1; j++) {
                        if (i == 1 && j == 1)
                            flags[it] = omp_is_initial_device();
                        b[IDX(i, j, n)] =
                            0.25 * (a[IDX(i - 1, j, n)] + a[IDX(i + 1, j, n)] +
                                    a[IDX(i, j - 1, n)] + a[IDX(i, j + 1, n)]);
                    }
            } else {
#pragma omp target teams distribute parallel for collapse(2) map(alloc : a[0:cells], b[0:cells], flags[0:flag_count])
                for (int i = 1; i < n - 1; i++)
                    for (int j = 1; j < n - 1; j++) {
                        if (i == 1 && j == 1)
                            flags[it] = omp_is_initial_device();
                        a[IDX(i, j, n)] =
                            0.25 * (b[IDX(i - 1, j, n)] + b[IDX(i + 1, j, n)] +
                                    b[IDX(i, j - 1, n)] + b[IDX(i, j + 1, n)]);
                    }
            }
            t.kernel_s += heat_wtime() - t0;
            t.iteracoes = it + 1;

            /* A convergência é opcional e não participa das campanhas fixas. */
            if (tol > 0.0 && t.iteracoes % 100 == 0) {
                double tv = heat_wtime();
                if (t.iteracoes & 1) {
#pragma omp target update from(b[0:cells])
                } else {
#pragma omp target update from(a[0:cells])
                }
                t.verificacoes++;
                const double *atual = (t.iteracoes & 1) ? b : a;
                double residuo = heat_residuo(atual, n);
                t.verificacao_s += heat_wtime() - tv;
                if (residuo < tol) break;
            }
        }

        double tr = heat_wtime();
        if (t.iteracoes & 1) {
#pragma omp target update from(b[0:cells])
        } else {
#pragma omp target update from(a[0:cells])
        }
        if (t.iteracoes > 0) {
#pragma omp target update from(flags[0:t.iteracoes])
        }
        t.saida_s = heat_wtime() - tr;
    }
    t.total_s = heat_wtime() - start;
    t.kernel_offload = 1;
    for (int it = 0; it < t.iteracoes; it++)
        if (flags[it] != 0) t.kernel_offload = 0;
    return t;
}

static void uso(const char *prog)
{
    fprintf(stderr,
            "Uso: %s [-n N] [-i ITER] [-c CASO] [-t TOL] [-o ARQ] [-a] [-H]\n"
            "  -n N     lado da grade (N x N)                         [1024]\n"
            "  -i ITER  número máximo de iterações                    [1000]\n"
            "  -c CASO  0=placa 1=uniforme 2=linear 3=ruído           [0]\n"
            "  -t TOL   verifica convergência a cada 100 passos\n"
            "  -o ARQ   grava a grade final em binário\n"
            "  -a       compara com a solução analítica após convergir\n"
            "  -H       usa o caminho host-only (somente validação local)\n",
            prog);
}

static int gravar_grade(const char *caminho, const double *u, size_t cells)
{
    FILE *f = fopen(caminho, "wb");
    if (!f) {
        perror(caminho);
        return 0;
    }
    int ok = fwrite(u, sizeof(*u), cells, f) == cells;
    if (fclose(f) != 0) ok = 0;
    if (!ok) fprintf(stderr, "Erro ao gravar a grade em %s\n", caminho);
    return ok;
}

static int validar(const double *u, int n, int caso, int analitica,
                   double tol, double residuo, int iteracoes)
{
    int ok = 1;
    HeatStats s = heat_estatisticas(u, n);
    int valores_finitos = 1;
    for (size_t k = 0; k < (size_t)n * (size_t)n; k++)
        if (!isfinite(u[k])) valores_finitos = 0;
    printf("todos os valores finitos              : %s\n",
           valores_finitos ? "OK" : "FALHOU");
    ok &= valores_finitos;
    int principio_maximo = s.minimo >= T_FRIO && s.maximo <= T_QUENTE;
    printf("princípio do máximo (0 <= T <= 100): %s\n",
           principio_maximo ? "OK" : "FALHOU");
    ok &= principio_maximo;

    if (caso == CASO_UNIFORME || caso == CASO_LINEAR) {
        double erro = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double d = fabs(u[IDX(i, j, n)] - heat_valor_inicial(caso, i, j, n));
                if (d > erro) erro = d;
            }
        int ponto_fixo = erro < 1e-10;
        printf("ponto fixo: erro máximo = %.3e       : %s\n",
               erro, ponto_fixo ? "OK" : "FALHOU");
        ok &= ponto_fixo;
    }

    if (analitica) {
        if (caso != CASO_PLACA || n > 1025) {
            printf("-a ignorado (somente caso 0 com N <= 1025)\n");
        } else {
            int convergiu = tol > 0.0 && residuo < tol && iteracoes % 100 == 0;
            double erro_centro = fabs(s.centro - 0.25 * T_QUENTE);
            int centro_ok = erro_centro < 1e-4;
            double soma2 = 0.0;
            for (int i = 1; i < n - 1; i++)
                for (int j = 1; j < n - 1; j++) {
                    double e = fabs(u[IDX(i, j, n)] - heat_analitica(i, j, n));
                    soma2 += e * e;
                }
            double rms = sqrt(soma2 / ((double)(n - 2) * (double)(n - 2)));
            int analitica_ok = rms < 0.5;
            printf("convergência (resíduo < tol)          : %s\n", convergiu ? "OK" : "FALHOU");
            printf("centro = %.8f (esperado 25) erro %.2e : %s\n",
                   s.centro, erro_centro, centro_ok ? "OK" : "FALHOU");
            printf("erro RMS vs. solução analítica = %.3e : %s\n",
                   rms, analitica_ok ? "OK" : "FALHOU");
            ok &= convergiu && centro_ok && analitica_ok;
        }
    }
    printf("VALIDACAO: %s\n", ok ? "OK" : "FALHOU");
    return ok;
}

int main(int argc, char **argv)
{
    int n = 1024, max_iter = 1000, caso = CASO_PLACA;
    int analitica = 0, permitir_host = 0, opt;
    double tol = 0.0;
    const char *arquivo = NULL;

    while ((opt = getopt(argc, argv, "n:i:c:t:o:aHh")) != -1) {
        switch (opt) {
        case 'n': n = atoi(optarg); break;
        case 'i': max_iter = atoi(optarg); break;
        case 'c': caso = atoi(optarg); break;
        case 't': tol = atof(optarg); break;
        case 'o': arquivo = optarg; break;
        case 'a': analitica = 1; break;
        case 'H': permitir_host = 1; break;
        case 'h': uso(argv[0]); return 0;
        default: uso(argv[0]); return 1;
        }
    }
    if (optind != argc || n < 3 || max_iter < 0 || caso < 0 || caso > 3 ||
        !isfinite(tol) || tol < 0.0 ||
        (size_t)n > SIZE_MAX / (size_t)n / sizeof(double)) {
        uso(argv[0]);
        return 1;
    }

    int dispositivos = omp_get_num_devices();
    int execucao_inicial = 1;
#pragma omp target map(from : execucao_inicial)
    { execucao_inicial = omp_is_initial_device(); }
    int offload_real = !execucao_inicial;
    printf("dispositivos_target = %d\n", dispositivos);
    printf("offload_real        = %d\n", offload_real);
    if (!offload_real && !permitir_host) {
        fprintf(stderr, "ERRO: região target executou no host; GPU real é obrigatória.\n");
        return 3;
    }
    if (!offload_real)
        fprintf(stderr, "AVISO: usando caminho host-only; este resultado não mede GPU.\n");

    size_t cells = (size_t)n * (size_t)n;
    size_t bytes_grid = cells * sizeof(double);
    double *a = heat_aloca(n);
    double *b = heat_aloca(n);
    size_t flag_count = max_iter > 0 ? (size_t)max_iter : 1;
    int *flags = (int *)calloc(flag_count, sizeof(*flags));
    if (!flags) {
        fprintf(stderr, "Erro: sem memória para flags de offload\n");
        free(a);
        free(b);
        return 1;
    }
    for (size_t k = 0; k < flag_count; k++) flags[k] = -1;
    heat_inicializa(a, n, caso);
    memcpy(b, a, bytes_grid); /* as bordas fixas precisam existir nos dois buffers */

    HeatTiming timing = offload_real
                            ? heat_evolui_device(a, b, flags, flag_count, n, cells,
                                                 max_iter, tol)
                            : heat_evolui_host(a, b, n, max_iter, tol);
    if (offload_real && !timing.kernel_offload) {
        fprintf(stderr, "ERRO: pelo menos um kernel target executou no host.\n");
        free(flags);
        free(a);
        free(b);
        return 3;
    }
    int iteracoes = timing.iteracoes;
    double *resultado = (iteracoes & 1) ? b : a;
    double residuo = heat_residuo(resultado, n);
    HeatStats s = heat_estatisticas(resultado, n);
    double mpoints = (double)(n - 2) * (double)(n - 2) * (double)iteracoes /
                     (timing.total_s > 0.0 ? timing.total_s : 1e-9) / 1e6;
    unsigned long long bytes_h2d = offload_real
                                       ? 2ULL * (unsigned long long)bytes_grid +
                                             (unsigned long long)flag_count * sizeof(*flags)
                                       : 0;
    unsigned long long bytes_d2h = offload_real
                                       ? (unsigned long long)(timing.verificacoes + 1) *
                                             (unsigned long long)bytes_grid +
                                             (unsigned long long)iteracoes * sizeof(*flags)
                                       : 0;

    printf("== heat_target ==\n");
    printf("N               = %d\n", n);
    printf("caso            = %d\n", caso);
    printf("iteracoes       = %d\n", iteracoes);
    printf("tempo_total_s   = %.9f\n", timing.total_s);
    printf("tempo_kernel_s  = %.9f\n", timing.kernel_s);
    printf("tempo_verif_s   = %.9f\n", timing.verificacao_s);
    printf("tempo_entrada_s = %.9f\n", timing.entrada_s);
    printf("tempo_saida_s   = %.9f\n", timing.saida_s);
    printf("mpontos_por_s   = %.3f\n", mpoints);
    printf("soma            = %.15e\n", s.soma);
    printf("norma_L2        = %.15e\n", s.norma_l2);
    printf("minimo          = %.15e\n", s.minimo);
    printf("maximo          = %.15e\n", s.maximo);
    printf("centro          = %.15e\n", s.centro);
    printf("perto_da_borda  = %.15e\n", s.perto_borda);
    printf("residuo         = %.6e\n", residuo);
    printf("bytes_h2d_estimados = %llu\n", bytes_h2d);
    printf("bytes_d2h_estimados = %llu\n", bytes_d2h);
    printf("CSV,target,%d,%d,1,1,%.9f\n", n, iteracoes, timing.total_s);
    printf("DET,target,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%llu,%llu,%d,%d\n",
           n, iteracoes, timing.total_s, timing.kernel_s, timing.verificacao_s,
           timing.entrada_s, timing.saida_s,
           bytes_h2d, bytes_d2h, offload_real, dispositivos);

    int output_ok = !arquivo || gravar_grade(arquivo, resultado, cells);
    int validation_ok = validar(resultado, n, caso, analitica, tol, residuo, iteracoes);
    free(flags);
    free(a);
    free(b);
    return output_ok && validation_ok ? 0 : 2;
}
