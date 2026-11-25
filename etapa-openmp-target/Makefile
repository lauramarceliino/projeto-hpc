# Perfis separados evitam confundir execução host-only com offloading real.
PROFILE ?= host
GCC ?= gcc
NVCC ?= nvc
SEQDIR = ../etapa-sequencial
BUILD = build
LDLIBS = -lm

ifeq ($(PROFILE),host)
CC = $(GCC)
CPUFLAGS = -std=c11 -O2 -Wall -Wextra -Werror
OMPFLAGS = -fopenmp -foffload=disable
else ifeq ($(PROFILE),nvhpc)
CC = $(NVCC)
CPUFLAGS = -c11 -O2 -Kieee
OMPFLAGS = -mp=gpu -gpu=ccnative,mem:separate,nofma -Minfo=mp
else
$(error PROFILE deve ser host ou nvhpc)
endif

.PHONY: all clean test test-host test-device info

all: $(BUILD)/heat_target $(BUILD)/heat_seq $(BUILD)/compare_grids

$(BUILD):
	mkdir -p $@

$(BUILD)/heat_target: heat_target.c $(SEQDIR)/heat_common.h | $(BUILD)
	$(CC) $(CPUFLAGS) $(OMPFLAGS) -o $@ $< $(LDLIBS)

$(BUILD)/heat_seq: $(SEQDIR)/heat_seq.c $(SEQDIR)/heat_common.h | $(BUILD)
	$(CC) $(CPUFLAGS) -I$(SEQDIR) -o $@ $< $(LDLIBS)

$(BUILD)/compare_grids: $(SEQDIR)/compare_grids.c | $(BUILD)
	$(CC) -O2 -Wall -Wextra -o $@ $< $(LDLIBS)

# Teste local: o perfil host desabilita offload e -H torna a intenção explícita.
test: test-host

test-host: all
	@echo "=== fallback local: deve ser identificado como offload_real=0 ==="
	@OMP_NUM_THREADS=1 ./$(BUILD)/heat_target -H -n 64 -i 20 -c 1 > $(BUILD)/host.log
	@grep 'offload_real' $(BUILD)/host.log | grep '= 0'
	@grep 'VALIDACAO: OK' $(BUILD)/host.log
	@echo "=== ponto fixo uniforme e linear ==="
	@OMP_NUM_THREADS=1 ./$(BUILD)/heat_target -H -n 64 -i 20 -c 1 > $(BUILD)/uniforme.log
	@grep 'VALIDACAO: OK' $(BUILD)/uniforme.log
	@OMP_NUM_THREADS=1 ./$(BUILD)/heat_target -H -n 64 -i 20 -c 2 > $(BUILD)/linear.log
	@grep 'VALIDACAO: OK' $(BUILD)/linear.log
	@echo "=== comparação grade irregular/ruído ==="
	@./$(BUILD)/heat_seq -n 47 -i 211 -c 3 -o $(BUILD)/seq.bin > /dev/null
	@OMP_NUM_THREADS=1 ./$(BUILD)/heat_target -H -n 47 -i 211 -c 3 -o $(BUILD)/target.bin > /dev/null
	@./$(BUILD)/compare_grids $(BUILD)/seq.bin $(BUILD)/target.bin 0
	@echo "TESTES HOST-ONLY: OK (não comprovam execução na GPU)"

# Este alvo deve ser executado em alocação GPU com OMP_TARGET_OFFLOAD=MANDATORY.
test-device: all
	@echo "=== offload real e casos fixos ==="
	@OMP_NUM_THREADS=1 ./$(BUILD)/heat_target -n 64 -i 20 -c 1 | grep 'VALIDACAO: OK'
	@OMP_NUM_THREADS=1 ./$(BUILD)/heat_target -n 64 -i 20 -c 2 | grep 'VALIDACAO: OK'
	@echo "=== comparação grade irregular/ruído ==="
	@./$(BUILD)/heat_seq -n 47 -i 211 -c 3 -o $(BUILD)/seq.bin > /dev/null
	@OMP_NUM_THREADS=1 ./$(BUILD)/heat_target -n 47 -i 211 -c 3 -o $(BUILD)/target.bin > /dev/null
	@./$(BUILD)/compare_grids $(BUILD)/seq.bin $(BUILD)/target.bin 0
	@echo "TESTES COM DISPOSITIVO: OK"

info:
	@echo "PROFILE=$(PROFILE) CC=$(CC)"
	@echo "CPUFLAGS=$(CPUFLAGS) OMPFLAGS=$(OMPFLAGS)"
	@$(CC) --version | head -n 1

clean:
	rm -f $(BUILD)/heat_target $(BUILD)/heat_seq $(BUILD)/compare_grids
	rm -f $(BUILD)/*.bin
	rmdir $(BUILD) 2>/dev/null || true
