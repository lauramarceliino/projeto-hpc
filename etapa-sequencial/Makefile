# Use as MESMAS flags de otimização em todas as versões (seq, MPI, OpenMP, GPU),
# senão o speedup fica distorcido. NÃO use -ffast-math: ele reordena somas e
# quebra a igualdade bit a bit entre versões.
CC      = gcc
CFLAGS  = -O3 -Wall -Wextra
LDLIBS  = -lm

all: heat_seq compare_grids

heat_seq: heat_seq.c heat_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# Quando existir a versão OpenMP (o run.slurm usa o alvo heat_<VERSAO>):
# heat_omp: heat_omp.c heat_common.h
# 	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDLIBS)

compare_grids: compare_grids.c
	$(CC) -O2 -Wall -Wextra -o $@ $< $(LDLIBS)

# Auto-testes do baseline (cada um termina com código != 0 se falhar)
test: all
	@echo "=== 1) uniforme: tudo deve continuar a 100 ==="
	./heat_seq -n 64 -i 500 -c 1
	@echo "=== 2) linear: ponto fixo, a grade nao pode mudar ==="
	./heat_seq -n 64 -i 500 -c 2
	@echo "=== 3) placa: convergir e comparar com a solucao analitica ==="
	./heat_seq -n 65 -i 500000 -t 1e-10 -a
	@echo "=== 4) determinismo: duas execucoes do caso 3 (ruido) geram grades identicas ==="
	@echo "    (mesmo procedimento para comparar a versao sequencial com as paralelas)"
	./heat_seq -n 128 -i 300 -c 3 -o ruido_a.bin > /dev/null
	./heat_seq -n 128 -i 300 -c 3 -o ruido_b.bin > /dev/null
	./compare_grids ruido_a.bin ruido_b.bin 0

clean:
	rm -f heat_seq compare_grids *.bin

.PHONY: all test clean
