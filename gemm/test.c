#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#if defined(__ARM_FEATURE_SME)
#include <arm_sme.h>
#endif

#if defined(__ARM_FEATURE_SME2)
#include <arm_sme.h>
#endif

#define SUCCESS 0
#define FAILURE 1
#define DEFAULT_N 128
#define EPSILON 1e-2 // 1e-2 = 0.01

static double get_time(clock_t start, clock_t end) {
    return (double)(end - start) / CLOCKS_PER_SEC;
}

void transpose(int N, const float *A, float *A_T) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            A_T[k * N + i] = A[i * N + k];
}

void gemm_naive(int N, const float *A, const float *B, float *C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

#if defined(__ARM_FEATURE_SVE)
void gemm_sve_basic(int N, const float *A, const float *B, float *C) {
    for (int i = 0; i < N; ++i) {
        const float *Ai = A + (size_t)i * (size_t)N;
        float *Ci = C + (size_t)i * (size_t)N;
        for (int j = 0; j < N; j += svcntw()) { //svcntw(): SVE CouNT Words (SVE 벡터 레지스터에 float이 몇개나 들어가는지,현재 16 (512bit))
            
            // svwhilelt_b32: WHILE Less Than (32-bit 타입용)
            // j부터 N까지 루프를 돌 때, 유효한 데이터 범위에 대한 '마스크(pg)'를 생성
            // 만약 남은 원소가 벡터 길이보다 적으면, 나머지 부분은 false로 채웁
            svbool_t pg = svwhilelt_b32((uint64_t)j, (uint64_t)N);

            // svld1: SVE LoaD (Vector 1)
            // pg(predicate)가 참인 부분만 Ci[j] 메모리에서 데이터를 읽어 벡터 레지스터로 Load
            svfloat32_t acc = svld1(pg, &Ci[j]);

            for (int k = 0; k < N; ++k) {
                // svdup_f32: SVE DUPlicate (float 32)
                // 스칼라 값 Ai[k]를 벡터의 모든 채널(lane)에 broadcast
                svfloat32_t a = svdup_f32(Ai[k]);

                // B 행렬의 k행 j열부터 데이터를 벡터로 로드
                svfloat32_t b = svld1(pg, &B[(size_t)k * (size_t)N + (size_t)j]);

                // svmla_f32_m: SVE Multiply-Add (float 32, Merging mode)
                // acc = acc + (a * b) 연산 수행
                // '_m'은 마스크(pg)가 false인 Lane에 대해 기존 acc 값을 그대로 '유지(Merge)'
                acc = svmla_f32_m(pg, acc, a, b);
            }

            // svst1: SVE STore (Vector 1)
            // 계산된 벡터 acc를 메모리 Ci[j]에 다시 저장. 마스크를 써서 안전하게 저장.
            svst1(pg, &Ci[j], acc);
        }
    }
}

void gemm_sve_unrolling(int N, const float *A, const float *B, float *C) {
    const int vl = svcntw();

    for (int i = 0; i < N; ++i) {
        const float *Ai = A + (size_t)i * (size_t)N;
        float *Ci = C + (size_t)i * (size_t)N;

        // 4배 언롤링: j를 vl * 4씩 증가
        int j = 0;
        for (; j <= N - (vl * 4); j += (vl * 4)) {
            // acc 4개 사용
            svfloat32_t acc0 = svld1(svptrue_b32(), &Ci[j]);
            svfloat32_t acc1 = svld1(svptrue_b32(), &Ci[j + vl]);
            svfloat32_t acc2 = svld1(svptrue_b32(), &Ci[j + vl * 2]);
            svfloat32_t acc3 = svld1(svptrue_b32(), &Ci[j + vl * 3]);

            for (int k = 0; k < N; ++k) {
                svfloat32_t a = svdup_f32(Ai[k]);
                
                // B 행렬에서 4개 영역의 데이터를 로드
                svfloat32_t b0 = svld1(svptrue_b32(), &B[(size_t)k * N + j]);
                svfloat32_t b1 = svld1(svptrue_b32(), &B[(size_t)k * N + j + vl]);
                svfloat32_t b2 = svld1(svptrue_b32(), &B[(size_t)k * N + j + vl * 2]);
                svfloat32_t b3 = svld1(svptrue_b32(), &B[(size_t)k * N + j + vl * 3]);

                // 4개의 FMA(Fused Multiply-Add) 유닛을 병렬 사용
                acc0 = svmla_f32_x(svptrue_b32(), acc0, a, b0);
                acc1 = svmla_f32_x(svptrue_b32(), acc1, a, b1);
                acc2 = svmla_f32_x(svptrue_b32(), acc2, a, b2);
                acc3 = svmla_f32_x(svptrue_b32(), acc3, a, b3);
            }

            svst1(svptrue_b32(), &Ci[j], acc0);
            svst1(svptrue_b32(), &Ci[j + vl], acc1);
            svst1(svptrue_b32(), &Ci[j + vl * 2], acc2);
            svst1(svptrue_b32(), &Ci[j + vl * 3], acc3);
        }

        // 남은 부분(Tail case) 처리: 기존 방식과 동일하게 Predicate 사용
        for (; j < N; j += vl) {
            svbool_t pg = svwhilelt_b32((uint64_t)j, (uint64_t)N);
            svfloat32_t acc = svld1(pg, &Ci[j]);
            for (int k = 0; k < N; ++k) {
                svfloat32_t a = svdup_f32(Ai[k]);
                svfloat32_t b = svld1(pg, &B[(size_t)k * N + j]);
                acc = svmla_f32_m(pg, acc, a, b);
            }
            svst1(pg, &Ci[j], acc);
        }
    }
}
#endif

#if defined(__ARM_FEATURE_SME)
[[arm::new("za"), arm::locally_streaming]]
void gemm_sme_transpose(int N, const float *A_T, const float *B, float *C) {
    // SVE/SME에서 한 벡터에 들어가는 float32 lane 수 (= vector length / 32bit)
    // SME의 ZA tile은 VL x VL 크기를 가지므로 block 크기의 기준이 됨
    int VL = svcntw();

    for (int i = 0; i < N; i += VL) {
        for (int j = 0; j < N; j += VL) {
            // ZA tile 전체를 0으로 초기화
            svzero_za();
            
            for (int k = 0; k < N; k++) {
                // A_T는 transpose된 행렬
                // A_T[k][i ~ i+VL] = 원래 A[i ~ i+VL][k] → column access를 contiguous load로 바꿈
                svbool_t pg_a = svwhilelt_b32(i, N);
                svfloat32_t va = svld1_f32(pg_a, &A_T[k * N + i]);

                // B[k][j ~ j+VL]
                svbool_t pg_b = svwhilelt_b32(j, N);
                svfloat32_t vb = svld1_f32(pg_b, &B[k * N + j]); 

                // outerproduct
                // (VL x 1) vector ⊗ (1 x VL) vector → (VL x VL) matrix 생성 후 누적
                svmopa_za32_f32_m(0, pg_a, pg_b, va, vb);
            }

            // ZA tile에 완성된 C block이 있음
            // ZA는 row 단위로만 store 가능 (horizontal store)
            for (int r = 0; r < VL; r++) {
                // edge 처리: 실제 matrix 범위를 넘으면 종료
                if (i + r >= N) break;

                // column 방향 predicate (j block edge)
                svbool_t pg_c = svwhilelt_b32(j, N);
                // ZA tile의 r번째 row를 메모리에 저장
                // ZA[r][0:VL] → C[i+r][j:j+VL]
                svst1_hor_za32(0, r, pg_c, &C[(i + r) * N + j]);
            }
        }
    }
}

#define KB (64) // k blocking
[[arm::new("za"), arm::locally_streaming]]
void gemm_sme_packed(int N, const float *A, const float *B, float *C) {
    int VL = svcntw();

    // A를 담을 임시 buffer (k_block × VL)
    float A_pack[KB * VL];

    for (int i = 0; i < N; i += VL) {
        for (int j = 0; j < N; j += VL) {
            svzero_za();

            // k blocking
            for (int kb = 0; kb < N; kb += KB) {
                int k_end = (kb + KB < N) ? (kb + KB) : N;

                // -------- A mini-packing --------
                // A[i:i+VL][kb:k_end] → A_pack[k][lane]
                for (int k = kb; k < k_end; k++) {
                    for (int lane = 0; lane < VL; lane++) {
                        if (i + lane < N)
                            A_pack[(k - kb) * VL + lane] = A[(i + lane) * N + k];
                        else
                            A_pack[(k - kb) * VL + lane] = 0.0f;
                    }
                }
                // -------- compute --------
                for (int k = kb; k < k_end; k++) {
                    // packed A load (contiguous)
                    svbool_t pg_a = svptrue_b32();  // 이미 padding 했으므로 full active
                    svfloat32_t va = svld1_f32(pg_a, &A_pack[(k - kb) * VL]);

                    // B row load (edge 처리 필요)
                    svbool_t pg_b = svwhilelt_b32(j, N);
                    svfloat32_t vb = svld1_f32(pg_b, &B[k * N + j]);

                    svmopa_za32_f32_m(0, pg_a, pg_b, va, vb);
                }
            }

            // -------- store --------
            for (int r = 0; r < VL; r++) {
                if (i + r >= N) break;

                svbool_t pg_c = svwhilelt_b32(j, N);
                svst1_hor_za32(0, r, pg_c,
                    &C[(i + r) * N + j]);
            }
        }
    }
}

#define UNROLL 4   // k loop unrolling factor
[[arm::new("za"), arm::locally_streaming]]
void gemm_sme_packed_ab_unroll(int N, const float *A, const float *B, float *C) {
    int VL = svcntw();
    // A_pack: (k_block × VL)
    float A_pack[KB * VL];
    // B_pack: (k_block × VL)
    float B_pack[KB * VL];

    for (int i = 0; i < N; i += VL) {
        for (int j = 0; j < N; j += VL) {
            svzero_za();  // ZA = C tile accumulator
            for (int kb = 0; kb < N; kb += KB) {
                int k_end = (kb + KB < N) ? (kb + KB) : N;
                int k_len = k_end - kb;
                // -------------------------
                // A mini-packing
                // A[i:i+VL][k] → A_pack[k][lane]
                // -------------------------
                for (int k = 0; k < k_len; k++) {
                    int kk = kb + k;
                    for (int lane = 0; lane < VL; lane++) {
                        if (i + lane < N)
                            A_pack[k * VL + lane] = A[(i + lane) * N + kk];
                        else
                            A_pack[k * VL + lane] = 0.0f;
                    }
                }

                // -------------------------
                // B mini-packing
                // B[k][j:j+VL] → B_pack[k][lane]
                // -------------------------
                for (int k = 0; k < k_len; k++) {
                    int kk = kb + k;

                    for (int lane = 0; lane < VL; lane++) {
                        if (j + lane < N)
                            B_pack[k * VL + lane] = B[kk * N + (j + lane)];
                        else
                            B_pack[k * VL + lane] = 0.0f;
                    }
                }

                // -------------------------
                // compute (k unrolling)
                // -------------------------
                int k = 0;

                // unrolled loop
                for (; k + (UNROLL - 1) < k_len; k += UNROLL) {

                    svbool_t pg = svptrue_b32(); // packed → full active

                    // --- k ---
                    svfloat32_t va0 = svld1_f32(pg, &A_pack[(k + 0) * VL]);
                    svfloat32_t vb0 = svld1_f32(pg, &B_pack[(k + 0) * VL]);
                    svmopa_za32_f32_m(0, pg, pg, va0, vb0);

                    // --- k+1 ---
                    svfloat32_t va1 = svld1_f32(pg, &A_pack[(k + 1) * VL]);
                    svfloat32_t vb1 = svld1_f32(pg, &B_pack[(k + 1) * VL]);
                    svmopa_za32_f32_m(0, pg, pg, va1, vb1);

                    // --- k+2 ---
                    svfloat32_t va2 = svld1_f32(pg, &A_pack[(k + 2) * VL]);
                    svfloat32_t vb2 = svld1_f32(pg, &B_pack[(k + 2) * VL]);
                    svmopa_za32_f32_m(0, pg, pg, va2, vb2);

                    // --- k+3 ---
                    svfloat32_t va3 = svld1_f32(pg, &A_pack[(k + 3) * VL]);
                    svfloat32_t vb3 = svld1_f32(pg, &B_pack[(k + 3) * VL]);
                    svmopa_za32_f32_m(0, pg, pg, va3, vb3);
                }

                // remainder 처리
                for (; k < k_len; k++) {
                    svbool_t pg = svptrue_b32();

                    svfloat32_t va = svld1_f32(pg, &A_pack[k * VL]);
                    svfloat32_t vb = svld1_f32(pg, &B_pack[k * VL]);

                    svmopa_za32_f32_m(0, pg, pg, va, vb);
                }
            }

            // -------------------------
            // store ZA → C
            // -------------------------
            for (int r = 0; r < VL; r++) {
                if (i + r >= N) break;

                svbool_t pg_c = svwhilelt_b32(j, N);
                svst1_hor_za32(0, r, pg_c, &C[(i + r) * N + j]);
            }
        }
    }
}

[[arm::new("za"), arm::locally_streaming]]
void gemm_sme_transpose_2x2(int N, const float *A_T, const float *B, float *C) {
    int VL = svcntw();

    // i, j를 2*VL 단위로 이동
    for (int i = 0; i < N; i += 2 * VL) {
        for (int j = 0; j < N; j += 2 * VL) {
            // ZA tile 4개 초기화
            svzero_za();

            for (int k = 0; k < N; k++) {
                // -------- A load (2개 row block) --------
                svbool_t pg_a0 = svwhilelt_b32(i, N);
                svbool_t pg_a1 = svwhilelt_b32(i + VL, N);

                svfloat32_t va0 = svld1_f32(pg_a0, &A_T[k * N + i]);
                svfloat32_t va1 = svld1_f32(pg_a1, &A_T[k * N + (i + VL)]);

                // -------- B load (2개 col block) --------
                svbool_t pg_b0 = svwhilelt_b32(j, N);
                svbool_t pg_b1 = svwhilelt_b32(j + VL, N);

                svfloat32_t vb0 = svld1_f32(pg_b0, &B[k * N + j]);
                svfloat32_t vb1 = svld1_f32(pg_b1, &B[k * N + (j + VL)]);

                // -------- 4개 outer product --------
                svmopa_za32_f32_m(0, pg_a0, pg_b0, va0, vb0); //C00
                svmopa_za32_f32_m(1, pg_a0, pg_b1, va0, vb1); //C01
                svmopa_za32_f32_m(2, pg_a1, pg_b0, va1, vb0); //C10
                svmopa_za32_f32_m(3, pg_a1, pg_b1, va1, vb1); //C11
            }

            // -------- store --------
            for (int r = 0; r < VL; r++) {
                if (i + r < N) { //C00
                    svbool_t pg = svwhilelt_b32(j, N);
                    svst1_hor_za32(0, r, pg, &C[(i + r) * N + j]);
                }
                if (i + r < N) { //C01
                    svbool_t pg = svwhilelt_b32(j + VL, N);
                    svst1_hor_za32(1, r, pg, &C[(i + r) * N + (j + VL)]);
                }
                if (i + VL + r < N) { //C10
                    svbool_t pg = svwhilelt_b32(j, N);
                    svst1_hor_za32(2, r, pg, &C[(i + VL + r) * N + j]);
                }
                if (i + VL + r < N) { //C11
                    svbool_t pg = svwhilelt_b32(j + VL, N);
                    svst1_hor_za32(3, r, pg, &C[(i + VL + r) * N + (j + VL)]);
                }
            }
        }
    }
}
#endif 

#if defined(__ARM_FEATURE_SME2)
[[arm::new("za"), arm::locally_streaming]]
void gemm_sme2_transpose_2x2(int N, const float *A_T, const float *B, float *C) { //Multi-vector (vector group) operations
    int VL = svcntw();

    for (int i = 0; i < N; i += 2 * VL) {
        for (int j = 0; j < N; j += 2 * VL) {
            svzero_za();

            // [SME2] Predicate-as-Counter (PN 레지스터) 생성
            // 연속된 2 * VL 크기만큼의 메모리 로드를 통제하는 SME2 전용 카운터
            svcount_t pn_a = svwhilelt_c32((uint64_t)i, (uint64_t)N, 2);
            svcount_t pn_b = svwhilelt_c32((uint64_t)j, (uint64_t)N, 2);

            // 기존 svmopa 연산용 boolean predicate (SME1 호환 및 연산용 유지)
            svbool_t pg_a0 = svwhilelt_b32(i, N);
            svbool_t pg_a1 = svwhilelt_b32(i + VL, N);
            svbool_t pg_b0 = svwhilelt_b32(j, N);
            svbool_t pg_b1 = svwhilelt_b32(j + VL, N);

            for (int k = 0; k < N; k++) {
                // ------------------------------------------------------------------
                // [SME2] Multi-vector Load
                // 컴파일 시 SME2 고유 어셈블리 명령어인 `ld1w {z0.s, z1.s}, pn0/z, [x0]` 
                // 형태로 변환되어, 1회의 명령으로 2개의 벡터를 동시에 로드
                // ------------------------------------------------------------------
                svfloat32x2_t va_tuple = svld1_f32_x2(pn_a, &A_T[k * N + i]);
                svfloat32_t va0 = svget2(va_tuple, 0);
                svfloat32_t va1 = svget2(va_tuple, 1);

                svfloat32x2_t vb_tuple = svld1_f32_x2(pn_b, &B[k * N + j]);
                svfloat32_t vb0 = svget2(vb_tuple, 0);
                svfloat32_t vb1 = svget2(vb_tuple, 1);

                // -------- 4개 outer product --------
                // svfloat32x2_t type에 대한 native MOPA intrinsic은 발견못함.. 
                svmopa_za32_f32_m(0, pg_a0, pg_b0, va0, vb0); // C00
                svmopa_za32_f32_m(1, pg_a0, pg_b1, va0, vb1); // C01
                svmopa_za32_f32_m(2, pg_a1, pg_b0, va1, vb0); // C10
                svmopa_za32_f32_m(3, pg_a1, pg_b1, va1, vb1); // C11
            }

            // -------- store (기존 로직 유지) --------
            for (int r = 0; r < VL; r++) {
                if (i + r < N) {
                    svbool_t pg = svwhilelt_b32(j, N);
                    svst1_hor_za32(0, r, pg, &C[(i + r) * N + j]);
                }
                if (i + r < N) {
                    svbool_t pg = svwhilelt_b32(j + VL, N);
                    svst1_hor_za32(1, r, pg, &C[(i + r) * N + (j + VL)]);
                }
                if (i + VL + r < N) {
                    svbool_t pg = svwhilelt_b32(j, N);
                    svst1_hor_za32(2, r, pg, &C[(i + VL + r) * N + j]);
                }
                if (i + VL + r < N) {
                    svbool_t pg = svwhilelt_b32(j + VL, N);
                    svst1_hor_za32(3, r, pg, &C[(i + VL + r) * N + (j + VL)]);
                }
            }
        }
    }
}
#endif

int verify_matrix(int N, const float *ref, const float *test, const char *name) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(ref[i] - test[i]) > EPSILON) {
            printf("[FAIL] %s Mismatch at index %d: Expected %f, Got %f\n", name, i, ref[i], test[i]);
            return FAILURE;
        }
    }
    printf("[PASS] %s result is correct!\n", name);
    return SUCCESS;
 }


int main(int argc, char *argv[]) {
    int N = DEFAULT_N;

    if (argc >= 2) {
        char *end;
        long n = strtol(argv[1], &end, 10);
        if (end != argv[1] && *end == '\0' && n > 0 && n <= INT_MAX)
            N = (int)n;
    }

    srand(time(NULL));

    size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    // Input array
    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *B_T = (float *)malloc(bytes);

    // Result array
    float *C_base = (float *)calloc((size_t)N * (size_t)N, sizeof(float));
    float *C_SVE = (float *)calloc((size_t)N * (size_t)N, sizeof(float));
    float *C_SME = (float *)calloc((size_t)N * (size_t)N, sizeof(float));
    float *C_SME2 = (float *)calloc((size_t)N * (size_t)N, sizeof(float));
    
    // Value initialization (0.0 ~ 1.0)
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / (float)RAND_MAX;
        B[i] = (float)rand() / (float)RAND_MAX;
    }

    printf("Matrix Size: %d x %d\n", N, N);
    printf("----------------------------------\n");

    clock_t start, end;

    // 1. Naive GEMM (answer sheet)
    start = clock();
    gemm_naive(N, A, B, C_base);
    end = clock();
    printf("1. Baseline (Core) Time : %.4f seconds\n", get_time(start, end));

    // 2. SVE
#if defined(__ARM_FEATURE_SVE)
    start = clock();
    //gemm_sve_basic(N, A, B, C_SVE);
    gemm_sve_unrolling(N, A, B, C_SVE);
    end = clock();
    printf("2. SVE Time : %.4f seconds\n", get_time(start, end));
    verify_matrix(N, C_base, C_SVE, "SVE");
#endif

    // 3. SME
#if defined(__ARM_FEATURE_SME)
    float *A_T = (float *)malloc(bytes);
    transpose(N, A, A_T);
    start = clock();
    gemm_sme_transpose(N, A_T, B, C_SME);
    //gemm_sme_packed(N, A, B, C_SME);
    //gemm_sme_packed_ab_unroll(N, A, B, C_SME);
    end = clock();
    printf("3. SME Time : %.4f seconds\n", get_time(start, end));
    verify_matrix(N, C_base, C_SME, "SME");
#endif

    // 4. SME2
#if defined(__ARM_FEATURE_SME2)
    float *A_T2 = (float *)malloc(bytes);
    transpose(N, A, A_T2);
    start = clock();
    gemm_sme2_transpose_2x2(N, A_T2, B, C_SME2);
    end = clock();
    printf("4. SME2 Time : %.4f seconds\n", get_time(start, end));
    verify_matrix(N, C_base, C_SME2, "SME2");
#endif

    free(A); free(B); free(B_T);
    free(C_base); free(C_SVE); free(C_SME); free(C_SME2);
    return 0;
}
