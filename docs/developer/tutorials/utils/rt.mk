RT_SRCS = ../utils/init.c ../utils/crt0.S ../utils/alloc.c ../utils/io.c ../utils/prf.c ../utils/string.c ../utils/fprintf.c
# Use picolibc headers/libs provided by Ubuntu's riscv64-unknown-elf packages.
PICO_SYSROOT ?= /usr/lib/picolibc/riscv64-unknown-elf
RT_FLAGS = -march=rv64imafdc -O3 -fno-tree-loop-distribute-patterns -I../utils -T../utils/link.ld -nostartfiles -nostdlib \
	--sysroot=$(PICO_SYSROOT) -specs=picolibc.specs
