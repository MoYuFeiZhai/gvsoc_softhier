#include <stdint.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"

#define MATMUL_M 8
#define MATMUL_N 8
#define MATMUL_K 8

#define L1_A_OFFSET 0x0000
#define L1_B_OFFSET 0x0800
#define L1_C_OFFSET 0x1000
#define L1_R_OFFSET 0x1800

static void init_and_ref(uint16_t *a, uint16_t *b, uint16_t *c, uint16_t *ref)
{
    // Initialize small deterministic matrices and a reference output in L1
    for (uint32_t i = 0; i < MATMUL_M; ++i) {
        for (uint32_t k = 0; k < MATMUL_K; ++k) {
            a[i * MATMUL_K + k] = (uint16_t)((i + 1) + (k % 3));
        }
    }

    for (uint32_t k = 0; k < MATMUL_K; ++k) {
        for (uint32_t j = 0; j < MATMUL_N; ++j) {
            b[k * MATMUL_N + j] = (uint16_t)((j + 1) + (k % 5));
        }
    }

    for (uint32_t i = 0; i < MATMUL_M * MATMUL_N; ++i) {
        c[i] = 0;
        ref[i] = 0;
    }

    // Simple reference GEMM on the host core for correctness checking
    for (uint32_t i = 0; i < MATMUL_M; ++i) {
        for (uint32_t j = 0; j < MATMUL_N; ++j) {
            uint32_t acc = 0;
            for (uint32_t k = 0; k < MATMUL_K; ++k) {
                acc += (uint32_t)a[i * MATMUL_K + k] * (uint32_t)b[k * MATMUL_N + j];
            }
            ref[i * MATMUL_N + j] = (uint16_t)acc;
        }
    }
}

int main(void)
{
    uint32_t eoc_val = 0;

    flex_barrier_xy_init();
    flex_global_barrier_xy();

    if (flex_get_cluster_id() == 0 && flex_is_first_core()) {
        // Use fixed L1 offsets to keep the testcase self-contained
        uint16_t *a = (uint16_t *)local(L1_A_OFFSET);
        uint16_t *b = (uint16_t *)local(L1_B_OFFSET);
        uint16_t *c = (uint16_t *)local(L1_C_OFFSET);
        uint16_t *ref = (uint16_t *)local(L1_R_OFFSET);

        init_and_ref(a, b, c, ref);

        printf("[RedMule] MatMul %dx%dx%d (uint16)\n", MATMUL_M, MATMUL_N, MATMUL_K);

        // Configure and trigger a single RedMule matmul in L1
        flex_redmule_config(MATMUL_M, MATMUL_N, MATMUL_K);
        flex_redmule_trigger((uint32_t)a, (uint32_t)b, (uint32_t)c, REDMULE_UINT_16);
        flex_redmule_wait();

        // Compare the RedMule result against the reference
        for (uint32_t i = 0; i < MATMUL_M * MATMUL_N; ++i) {
            if (c[i] != ref[i]) {
                uint32_t row = i / MATMUL_N;
                uint32_t col = i % MATMUL_N;
                printf("[Mismatch] (%u,%u) got=%u expected=%u\n", row, col, c[i], ref[i]);
                eoc_val = 1;
                break;
            }
        }

        if (eoc_val == 0) {
            printf("[OK] RedMule MatMul matches reference\n");
        }
    }

    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
