MPICC    ?= mpicc
CFLAGS    = -O3 -Wall -Wextra -fopenmp
CPPFLAGS  = -I$(SEQDIR)
LDLIBS    = -lm
SEQDIR    = ../etapa-sequencial
SEQ       = $(SEQDIR)/heat_seq
CMP       = $(SEQDIR)/compare_grids

MPIRUN       ?= mpirun
MPIRUN_FLAGS ?=
RUN           = $(MPIRUN) $(MPIRUN_FLAGS)
SIG           = ^(soma|norma_L2|minimo|maximo|centro|perto_da_borda|residuo)

all: heat_hib

heat_hib: heat_hib.c $(SEQDIR)/heat_common.h
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

seq:
	$(MAKE) -C $(SEQDIR) heat_seq compare_grids

# Configurações do teste 4: N:processos:threads
CONFS = 128:1:1 128:3:2 128:7:2 131:5:1 131:6:2 10:8:1

test: heat_hib seq
	@echo "=== 1) uniforme: tudo deve continuar a 100 ==="
	OMP_NUM_THREADS=2 $(RUN) -n 2 ./heat_hib -n 64 -i 500 -c 1
	@echo "=== 2) linear: ponto fixo, a grade nao pode mudar ==="
	OMP_NUM_THREADS=2 $(RUN) -n 3 ./heat_hib -n 64 -i 500 -c 2
	@echo "=== 3) placa: convergir e comparar com a solucao analitica ==="
	OMP_NUM_THREADS=2 $(RUN) -n 4 ./heat_hib -n 65 -i 500000 -t 1e-10 -a
	@echo "=== 4) grade e assinatura identicas ao sequencial (caso 3, ruido) ==="
	@for cfg in $(CONFS); do \
	    n=$${cfg%%:*}; r=$${cfg#*:}; p=$${r%%:*}; t=$${r#*:}; \
	    $(SEQ) -n $$n -i 300 -c 3 -o seq.bin | grep -E "$(SIG)" > seq.sig; \
	    for b in "" -b; do \
	        OMP_NUM_THREADS=$$t $(RUN) -n $$p ./heat_hib -n $$n -i 300 -c 3 $$b -o hib.bin | grep -E "$(SIG)" > hib.sig; \
	        if $(CMP) seq.bin hib.bin 0 > /dev/null && cmp -s seq.sig hib.sig; then \
	            echo "  N=$$n P=$$p T=$$t $${b:-sobreposto}: OK"; \
	        else echo "  N=$$n P=$$p T=$$t $${b:-sobreposto}: FALHOU"; rm -f *.sig; exit 1; fi; \
	    done; \
	done; rm -f seq.bin hib.bin seq.sig hib.sig
	@echo "TESTES DA VERSAO HIBRIDA: OK"

clean:
	rm -f heat_hib *.bin *.sig

.PHONY: all seq test clean
