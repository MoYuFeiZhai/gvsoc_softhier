#include <stdint.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_dma_pattern.h"

#define MATMUL_M 8
#define MATMUL_N 8
#define MATMUL_K 8

#define L1_A_OFFSET 0x0000
#define L1_B_OFFSET 0x0800
#define L1_C_OFFSET 0x1000
#define L1_R_OFFSET 0x1800
#define L1_FLAG_OFFSET 0x2000

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

    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();

    if (flex_is_first_core() && (cluster_id == 0 || cluster_id == 1)) {
        // Use fixed L1 offsets to keep the testcase self-contained
        uint16_t *a = (uint16_t *)local(L1_A_OFFSET);
        uint16_t *b = (uint16_t *)local(L1_B_OFFSET);
        uint16_t *c = (uint16_t *)local(L1_C_OFFSET);
        uint16_t *ref = (uint16_t *)local(L1_R_OFFSET);
        uint32_t *flag = (uint32_t *)local(L1_FLAG_OFFSET);

        init_and_ref(a, b, c, ref);
        *flag = 0;

        if (cluster_id == 0) {
            printf("[RedMule] MatMul %dx%dx%d (uint16)\n", MATMUL_M, MATMUL_N, MATMUL_K);

            // Configure and trigger a single RedMule matmul in L1
            flex_redmule_config(MATMUL_M, MATMUL_N, MATMUL_K);
            flex_redmule_trigger((uint32_t)a, (uint32_t)b, (uint32_t)c, REDMULE_UINT_16);
            flex_redmule_wait();
        }
    }

    flex_global_barrier_xy();

    // Cluster-to-cluster handoff via iDMA: cluster 0 -> cluster 1.
    if (flex_is_dm_core() && cluster_id == 0) {
        flex_dma_async_1d(
            remote_xy(1, 0, L1_C_OFFSET),
            local(L1_C_OFFSET),
            MATMUL_M * MATMUL_N * sizeof(uint16_t));
        flex_dma_async_wait_all();
    }

    flex_global_barrier_xy();

    // Cluster 1 verifies the received output against its local reference.
    if (flex_is_first_core() && cluster_id == 1) {
        uint16_t *c = (uint16_t *)local(L1_C_OFFSET);
        uint16_t *ref = (uint16_t *)local(L1_R_OFFSET);
        uint32_t *flag = (uint32_t *)local(L1_FLAG_OFFSET);

        for (uint32_t i = 0; i < MATMUL_M * MATMUL_N; ++i) {
            if (c[i] != ref[i]) {
                uint32_t row = i / MATMUL_N;
                uint32_t col = i % MATMUL_N;
                printf("[Cluster1 Mismatch] (%u,%u) got=%u expected=%u\n", row, col, c[i], ref[i]);
                *flag = 1;
                break;
            }
        }
    }

    flex_global_barrier_xy();

    // Send cluster 1's result flag back to cluster 0 via iDMA.
    if (flex_is_dm_core() && cluster_id == 1) {
        flex_dma_async_1d(
            remote_xy(0, 0, L1_FLAG_OFFSET),
            local(L1_FLAG_OFFSET),
            sizeof(uint32_t));
        flex_dma_async_wait_all();
    }

    flex_global_barrier_xy();

    if (flex_is_first_core() && cluster_id == 0) {
        uint32_t *flag = (uint32_t *)local(L1_FLAG_OFFSET);
        if (*flag == 0) {
            printf("[OK] Cluster0 -> Cluster1 handoff matches reference\n");
        } else {
            eoc_val = 1;
        }
    }

    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
