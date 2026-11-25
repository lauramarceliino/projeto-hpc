/*
 * heat_seq.c — Difusão de calor 2D: versão SEQUENCIAL (baseline)
 *
 * Método: diferenças finitas, stencil de 5 pontos (iteração de Jacobi):
 *     T_novo[i][j] = 0,25 * (T[i-1][j] + T[i+1][j] + T[i][j-1] + T[i][j+1])
 * As bordas são fixas (condição de Dirichlet): só os pontos INTERNOS são atualizados.
 *
 * Uso:  ./heat_seq -n N -i ITER [-c CASO] [-t TOL] [-o ARQ.bin] [-a]   (completo, com validação)
 *       ./heat_seq N ITER                                                (modo PaScal, enxuto)
 *       (veja uso() abaixo)
 */
#include "heat_common.h"

#include <string.h>
#include <unistd.h> /* getopt */

/* Um passo de Jacobi: lê 'u' (tempo atual) e escreve 'un' (próximo tempo).
 *
 * - O laço interno percorre j (colunas), que é a direção contígua na memória.
 * - 'restrict' promete ao compilador que u e un não se sobrepõem (ajuda a vetorizar).
 * - Mantenha EXATAMENTE esta ordem de soma nas outras versões: assim cada célula
 *   é calculada com as mesmas operações e os resultados saem idênticos bit a bit. */
static void heat_passo(const double *restrict u, double *restrict un, int n)
{
    for (int i = 1; i < n - 1; i++)
        for (int j = 1; j < n - 1; j++)
            un[IDX(i, j, n)] = 0.25 * (u[IDX(i - 1, j, n)] + u[IDX(i + 1, j, n)] +
                                       u[IDX(i, j - 1, n)] + u[IDX(i, j + 1, n)]);
}

static void uso(const char *prog)
{
    fprintf(stderr,
            "Uso: %s [-n N] [-i ITER] [-c CASO] [-t TOL] [-o ARQ] [-a]\n"
            "  ou %s N ITER    (modo PaScal: enxuto, sem validação nem estatísticas)\n"
            "  -n N     lado da grade (N x N)                          [1024]\n"
            "  -i ITER  número (máximo) de iterações                   [1000]\n"
            "  -c CASO  0=placa 1=uniforme 2=linear 3=ruído            [0]\n"
            "  -t TOL   para antes se o resíduo < TOL (só validação)   [desligado]\n"
            "  -o ARQ   grava a grade final em binário (use N pequeno)\n"
            "  -a       compara com a solução analítica (caso 0; use com -t e N <= 1025)\n",
            prog, prog);
}

/* Confere o resultado. Devolve 1 se tudo passou. */
static int validar(const double *u, int n, int caso, const HeatStats *s,
                   int analitica, double tol, double residuo)
{
    int ok = 1;
    printf("-- validação --\n");

    /* (a) Princípio do máximo: média de valores em [0,100] continua em [0,100]. */
    int pm = (s->minimo >= T_FRIO && s->maximo <= T_QUENTE);
    printf("princípio do máximo (0 <= T <= 100)  : %s\n", pm ? "OK" : "FALHOU");
    ok &= pm;

    /* (b) Casos "ponto fixo": a grade final tem de ser igual à inicial. */
    if (caso == CASO_UNIFORME || caso == CASO_LINEAR) {
        double erro = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double d = fabs(u[IDX(i, j, n)] - heat_valor_inicial(caso, i, j, n));
                if (d > erro) erro = d;
            }
        int r = erro < 1e-10;
        printf("ponto fixo: erro máx. = %.3e          : %s\n", erro, r ? "OK" : "FALHOU");
        ok &= r;
    }

    /* (c) Placa, regime estacionário: comparar com a solução analítica. */
    if (analitica) {
        if (caso != CASO_PLACA || n > 1025) {
            printf("-a ignorado (só vale para o caso 0 com N <= 1025)\n");
        } else {
            int conv = (tol > 0.0 && residuo < tol);
            printf("convergiu (resíduo < tol)            : %s\n", conv ? "OK" : "NÃO (aumente -i)");

            /* Por simetria (4 rotações de 90° somam a borda inteira a 100 °C),
             * o centro vale exatamente 25 °C quando a solução converge. */
            double erro_centro = fabs(s->centro - 0.25 * T_QUENTE);
            int rc = erro_centro < 1e-4;
            printf("centro = %.8f (esperado 25)  erro = %.2e : %s\n", s->centro, erro_centro,
                   rc ? "OK" : "FALHOU");

            double soma2 = 0.0, emax = 0.0;
            for (int i = 1; i < n - 1; i++)
                for (int j = 1; j < n - 1; j++) {
                    double e = fabs(u[IDX(i, j, n)] - heat_analitica(i, j, n));
                    soma2 += e * e;
                    if (e > emax) emax = e;
                }
            double rms = sqrt(soma2 / ((double)(n - 2) * (double)(n - 2)));
            /* Erro de discretização (medido): o RMS cai ~2x cada vez que N dobra, e o máximo
             * fica ~0,7 °C, junto aos cantos quentes, onde o contorno é descontínuo.
             * O teste afiado é o do centro; este pega erros grosseiros (valores medidos
             * com N >= 33: RMS < 0,07; um stencil errado dá RMS de dezenas de °C). */
            int ra = rms < 0.5;
            printf("vs analítica: RMS = %.3e  máx = %.3e (°C) : %s\n", rms, emax,
                   ra ? "OK" : "FALHOU");
            ok &= conv && rc && ra;
        }
    }

    printf("VALIDACAO: %s\n", ok ? "OK" : "FALHOU");
    return ok;
}

int main(int argc, char **argv)
{
    int n = 1024, iters = 1000, caso = CASO_PLACA, analitica = 0, opt;
    double tol = 0.0;
    const char *arq_saida = NULL;

    while ((opt = getopt(argc, argv, "n:i:c:t:o:ah")) != -1) {
        switch (opt) {
        case 'n': n = atoi(optarg); break;
        case 'i': iters = atoi(optarg); break;
        case 'c': caso = atoi(optarg); break;
        case 't': tol = atof(optarg); break;
        case 'o': arq_saida = optarg; break;
        case 'a': analitica = 1; break;
        default: uso(argv[0]); return 1;
        }
    }
    /* Modo PaScal: argumentos POSICIONAIS  ./heat_seq N ITER
     * É assim que o PaScal Analyzer chama o programa (-i "4096 100" -> ./heat_seq 4096 100).
     * Neste modo o programa só inicializa, itera e imprime um checksum barato: o PaScal mede
     * o tempo do programa INTEIRO, então qualquer passada extra sobre a grade (resíduo,
     * estatísticas, validação) entraria na conta e viraria trecho serial nas versões paralelas. */
    int modo_pascal = (optind < argc);
    if (modo_pascal) {
        n = atoi(argv[optind]);
        if (optind + 1 < argc) iters = atoi(argv[optind + 1]);
    }
    if (n < 3 || iters < 0 || caso < 0 || caso > 3) {
        uso(argv[0]);
        return 1;
    }

    /* Dois buffers: Jacobi lê de um e escreve no outro. */
    double *u = heat_aloca(n);
    double *un = heat_aloca(n);

    heat_inicializa(u, n, caso);
    /* As bordas nunca são recalculadas, mas os ponteiros são trocados a cada
     * iteração: por isso a borda precisa existir nos DOIS buffers. */
    memcpy(un, u, (size_t)n * (size_t)n * sizeof(double));

    if (!modo_pascal) {
        printf("== heat_seq ==\n");
        printf("N               = %d (%.2f milhões de pontos, %.1f MB por buffer, 2 buffers)\n", n,
               (double)n * n / 1e6, (double)n * n * sizeof(double) / 1e6);
        printf("caso            = %d\n", caso);
    }

    /* ---- laço temporal: é a ÚNICA parte cronometrada ---- */
    double t0 = heat_wtime();
    int it = 0;
    for (; it < iters; it++) {
        heat_passo(u, un, n);
        double *tmp = u; u = un; un = tmp; /* troca de ponteiros: sem copiar a matriz */

        /* Critério de parada opcional (modo validação; não use ao medir desempenho) */
        if (tol > 0.0 && (it + 1) % 100 == 0 && heat_residuo(u, n) < tol) {
            it++;
            break;
        }
    }
    double tempo = heat_wtime() - t0;
    /* ------------------------------------------------------ */

    if (modo_pascal) {
        double traco = 0.0; /* custo O(N): basta para o compilador não eliminar o laço */
        for (int i = 0; i < n; i++) traco += u[IDX(i, i, n)];
        printf("N: %d | Iter: %d | Threads: 1 | Tempo: %.6f s | Trace: %.15e\n", n, it, tempo, traco);
        free(u);
        free(un);
        return 0;
    }

    HeatStats s = heat_estatisticas(u, n);
    double residuo = heat_residuo(u, n);
    double mpontos = (double)(n - 2) * (n - 2) * it / (tempo > 0 ? tempo : 1e-9) / 1e6;

    printf("iteracoes       = %d\n", it);
    printf("tempo_loop_s    = %.6f\n", tempo);
    printf("mpontos_por_s   = %.1f\n", mpontos);
    printf("soma            = %.15e\n", s.soma);
    printf("norma_L2        = %.15e\n", s.norma_l2);
    printf("minimo          = %.15e\n", s.minimo);
    printf("maximo          = %.15e\n", s.maximo);
    printf("centro          = %.15e\n", s.centro);
    printf("perto_da_borda  = %.15e\n", s.perto_borda);
    printf("residuo         = %.6e\n", residuo);
    /* versao,N,iteracoes,processos,threads,tempo_s  -> fácil de juntar num CSV:  grep ^CSV */
    printf("CSV,seq,%d,%d,1,1,%.6f\n", n, it, tempo);

    if (arq_saida) {
        FILE *f = fopen(arq_saida, "wb");
        if (!f || fwrite(u, sizeof(double), (size_t)n * (size_t)n, f) != (size_t)n * (size_t)n) {
            fprintf(stderr, "Erro ao gravar %s\n", arq_saida);
            return 1;
        }
        fclose(f);
        printf("grade gravada em %s\n", arq_saida);
    }

    int ok = validar(u, n, caso, &s, analitica, tol, residuo);

    free(u);
    free(un);
    return ok ? 0 : 2;
}
