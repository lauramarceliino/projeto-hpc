/*
 * compare_grids.c — compara duas grades gravadas em binário (double, row-major),
 * por exemplo a saída do heat_seq (-o) e a de uma versão paralela.
 *
 * Uso:  ./compare_grids a.bin b.bin [tolerancia]     (padrão: 1e-12)
 * Saída: código 0 se a maior diferença absoluta <= tolerância, 1 caso contrário.
 *
 * Como todas as versões fazem as mesmas operações em cada célula, o esperado é
 * diferença ZERO (bit a bit). Tolerância só é necessária se alguém usar
 * -ffast-math ou outra mudança que reordene as somas.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCO (1 << 16) /* doubles lidos por vez */

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Uso: %s a.bin b.bin [tolerancia=1e-12]\n", argv[0]);
        return 2;
    }
    double tol = (argc > 3) ? atof(argv[3]) : 1e-12;

    FILE *fa = fopen(argv[1], "rb");
    FILE *fb = fopen(argv[2], "rb");
    if (!fa || !fb) {
        fprintf(stderr, "Erro ao abrir os arquivos\n");
        return 2;
    }

    double *a = (double *)malloc(BLOCO * sizeof(double));
    double *b = (double *)malloc(BLOCO * sizeof(double));
    size_t total = 0, diferentes = 0, nans = 0, pos_max = 0;
    double max_dif = 0.0, soma2 = 0.0;

    for (;;) {
        size_t ra = fread(a, sizeof(double), BLOCO, fa);
        size_t rb = fread(b, sizeof(double), BLOCO, fb);
        if (ra != rb) {
            printf("RESULTADO: FALHOU (arquivos com tamanhos diferentes)\n");
            return 1;
        }
        if (ra == 0) break;
        for (size_t k = 0; k < ra; k++) {
            double d = fabs(a[k] - b[k]);
            if (isnan(d)) { nans++; continue; }
            if (d != 0.0) diferentes++;
            if (d > max_dif) { max_dif = d; pos_max = total + k; }
            soma2 += d * d;
        }
        total += ra;
    }

    size_t n = (size_t)(sqrt((double)total) + 0.5);
    printf("elementos            : %zu (grade %zu x %zu)\n", total, n, n);
    printf("diferentes (bit)     : %zu\n", diferentes);
    printf("NaN                  : %zu\n", nans);
    printf("maior |a-b|          : %.3e", max_dif);
    if (max_dif > 0.0 && n > 0) printf("  em (i=%zu, j=%zu)", pos_max / n, pos_max % n);
    printf("\nRMS da diferença     : %.3e\n", total ? sqrt(soma2 / (double)total) : 0.0);

    int ok = (nans == 0 && total > 0 && max_dif <= tol);
    printf("RESULTADO: %s (tol = %.1e)\n", ok ? "OK" : "FALHOU", tol);
    return ok ? 0 : 1;
}
