/*
 * heat_common.h — código COMPARTILHADO por todas as versões do projeto
 * (sequencial, MPI+OpenMP, OpenMP Target, CUDA).
 *
 * Contém:
 *   1. Gerador de dados sintéticos: valor inicial de QUALQUER célula global (i, j)
 *   2. Estatísticas ("assinatura") do resultado, para comparar versões entre si
 *   3. Resíduo (critério de convergência)
 *   4. Solução analítica (valida o baseline em grades pequenas)
 *
 * Tudo é 'static inline': basta um #include (funciona com gcc, mpicc e nvcc).
 */
#ifndef HEAT_COMMON_H
#define HEAT_COMMON_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define T_QUENTE 100.0 /* temperatura da borda quente (°C) */
#define T_FRIO     0.0 /* temperatura das demais bordas (°C) */

/* Matriz n x n guardada em um vetor 1D, linha após linha (row-major).
 * O cast para size_t evita overflow de int em grades grandes. */
#define IDX(i, j, n) ((size_t)(i) * (size_t)(n) + (size_t)(j))

/* Casos de teste do gerador sintético */
enum {
    CASO_PLACA    = 0, /* borda i=0 a 100 °C, resto 0 °C  <- o caso do projeto/benchmark */
    CASO_UNIFORME = 1, /* tudo a 100 °C: deve permanecer constante (ponto fixo)          */
    CASO_LINEAR   = 2, /* perfil linear em i e j: também é ponto fixo do método          */
    CASO_RUIDO    = 3  /* interior pseudoaleatório determinístico: pega erros de halo    */
};

/* ------------------------------------------------------------------ */
/* 1. GERADOR DE DADOS SINTÉTICOS                                      */
/* ------------------------------------------------------------------ */

/* Hash de inteiros (finalizador do splitmix64) -> número em [0, 1).
 * Não tem estado: o valor depende só do índice global da célula, então é igual
 * em qualquer processo, thread ou GPU, não importa como a grade foi dividida. */
static inline double heat_ruido(unsigned long long k)
{
    unsigned long long z = k + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0); /* 53 bits / 2^53 */
}

static inline int heat_eh_borda(int i, int j, int n)
{
    return i == 0 || j == 0 || i == n - 1 || j == n - 1;
}

/* Valor inicial da célula GLOBAL (i, j) de uma grade n x n.
 * Na versão MPI, cada rank chama esta função só para as linhas que lhe pertencem
 * (usando o índice global da linha): ninguém precisa montar nem espalhar a matriz. */
static inline double heat_valor_inicial(int caso, int i, int j, int n)
{
    switch (caso) {
    case CASO_UNIFORME:
        return T_QUENTE;
    case CASO_LINEAR: {
        double x = (double)j / (n - 1);
        double y = (double)i / (n - 1);
        return T_QUENTE * (x + 2.0 * y) / 3.0; /* declives diferentes em i e j */
    }
    case CASO_RUIDO:
        if (!heat_eh_borda(i, j, n))
            return T_QUENTE * heat_ruido(IDX(i, j, n));
        break; /* bordas: mesma regra da placa */
    default:
        break;
    }
    return (i == 0) ? T_QUENTE : T_FRIO; /* CASO_PLACA (e bordas do ruído) */
}

static inline void heat_inicializa(double *u, int n, int caso)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            u[IDX(i, j, n)] = heat_valor_inicial(caso, i, j, n);
}

static inline double *heat_aloca(int n)
{
    double *p = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!p) {
        fprintf(stderr, "Erro: sem memória para uma grade %d x %d\n", n, n);
        exit(1);
    }
    return p;
}

static inline double heat_wtime(void) /* tempo de parede, em segundos */
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
    /* Nas outras versões use omp_get_wtime() / MPI_Wtime() */
}

/* ------------------------------------------------------------------ */
/* 2. ESTATÍSTICAS ("assinatura" do resultado)                         */
/* ------------------------------------------------------------------ */

typedef struct {
    double soma, norma_l2, minimo, maximo;
    double centro;      /* centro da placa (média das 4 células centrais se n for par) */
    double perto_borda; /* u[1][n/2]: vizinho da borda quente, no eixo de simetria     */
} HeatStats;

static inline HeatStats heat_estatisticas(const double *u, int n)
{
    HeatStats s;
    double soma = 0.0, soma2 = 0.0, mn = u[0], mx = u[0];
    for (int i = 0; i < n; i++) {
        double ls = 0.0, ls2 = 0.0; /* soma parcial da linha: menos erro de arredondamento */
        for (int j = 0; j < n; j++) {
            double v = u[IDX(i, j, n)];
            ls += v;
            ls2 += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        soma += ls;
        soma2 += ls2;
    }
    s.soma = soma;
    s.norma_l2 = sqrt(soma2);
    s.minimo = mn;
    s.maximo = mx;
    int m = n / 2;
    if (n % 2)
        s.centro = u[IDX(m, m, n)];
    else
        s.centro = 0.25 * (u[IDX(m - 1, m - 1, n)] + u[IDX(m - 1, m, n)] +
                           u[IDX(m, m - 1, n)] + u[IDX(m, m, n)]);
    s.perto_borda = u[IDX(1, m, n)];
    return s;
}

/* ------------------------------------------------------------------ */
/* 3. RESÍDUO                                                          */
/* ------------------------------------------------------------------ */

/* max |média_dos_vizinhos - u| nos pontos internos = maior mudança que o
 * PRÓXIMO passo faria. Tende a zero quando a solução estaciona. */
static inline double heat_residuo(const double *u, int n)
{
    double r = 0.0;
    for (int i = 1; i < n - 1; i++)
        for (int j = 1; j < n - 1; j++) {
            double media = 0.25 * (u[IDX(i - 1, j, n)] + u[IDX(i + 1, j, n)] +
                                   u[IDX(i, j - 1, n)] + u[IDX(i, j + 1, n)]);
            double d = fabs(media - u[IDX(i, j, n)]);
            if (d > r) r = d;
        }
    return r;
}

/* ------------------------------------------------------------------ */
/* 4. SOLUÇÃO ANALÍTICA (regime estacionário, CASO_PLACA)              */
/* ------------------------------------------------------------------ */

/* Equação de Laplace no quadrado unitário com T = T_QUENTE na borda i = 0 e
 * T = 0 nas outras três. Série de Fourier (somente k ímpar):
 *
 *   T(x, s) = (4 T_QUENTE / pi) * SOMA (1/k) sin(k pi x) sinh(k pi (1-s)) / sinh(k pi)
 *
 * com x = j/(n-1) e s = i/(n-1) (distância até a borda quente).
 * A razão de senos hiperbólicos é reescrita com exponenciais negativas
 * para não estourar o double. Os termos decaem como exp(-k pi s): paramos
 * quando ficam abaixo de ~1e-17. Custo alto: use só em grades pequenas. */
static inline double heat_analitica(int i, int j, int n)
{
    const double PI = 3.14159265358979323846;
    if (i == 0) return T_QUENTE;
    if (i == n - 1 || j == 0 || j == n - 1) return T_FRIO;

    double x = (double)j / (n - 1);
    double s = (double)i / (n - 1);
    double soma = 0.0;
    for (int k = 1; k * PI * s < 40.0; k += 2) {
        double razao = (exp(-k * PI * s) - exp(-k * PI * (2.0 - s))) /
                       (1.0 - exp(-2.0 * k * PI));
        soma += sin(k * PI * x) / k * razao;
    }
    return 4.0 * T_QUENTE / PI * soma;
}

#endif /* HEAT_COMMON_H */
