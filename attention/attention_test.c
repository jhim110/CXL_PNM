#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

#define SUCCESS 0
#define FAILURE 1
#define EPSILON 1e-2 // 1e-2 = 0.01

// 데이터 타입 정의
typedef float target_dtype;


double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_matrix(target_dtype *A, int N) {
    for (int i = 0; i < N; i++) {
        A[i] = (target_dtype)((float)rand() / RAND_MAX);
    }
}

int verify_matrix(int N, const target_dtype *ref, const target_dtype *test, const char *name) {
    for (int i = 0; i < N; i++) {
        if (fabs(ref[i] - test[i]) > EPSILON) {
            printf("[FAIL] %s Mismatch at index %d: Expected %f, Got %f\n", name, i, ref[i], test[i]);
            return FAILURE;
        }
    }
    printf("[PASS] %s result is correct!\n", name);
    return SUCCESS;
 }

void transpose (int M, int N, target_dtype *A, target_dtype *B) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[j*M + i] = A[i*N + j];
        }
    }
 }

void gemm(int M, int N, int K, target_dtype *A, target_dtype *B, target_dtype *C) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        target_dtype sum = 0.0f;
          for (int k = 0; k < K; k++) {
              sum += A[i*K + k] * B[k*N + j];
          }
          C[i*N + j] = sum;
      }
    }
}

void scaling(target_dtype* s, int M, int N, int D) {
    target_dtype scale = 1.0f / sqrtf((target_dtype)D);
    for (int i = 0; i < M*N; i++) {
        s[i] *= scale;
    }
}

void softmax(int N, target_dtype *A) {
    for (int i = 0; i < N; i++) {
        target_dtype max = A[i*N];
        for (int j = 1; j < N; j++) {
            if (A[i*N + j] > max) max = A[i*N + j];
        }

        target_dtype sum = 0.0f;
        for (int j = 0; j < N; j++) {
            A[i*N + j] = expf(A[i*N + j] - max);
            sum += A[i*N + j];
        }

        for (int j = 0; j < N; j++) {
            A[i*N + j] /= sum;
        }
    }
}

// MHA (batched head)
void mha_batch(target_dtype *Q, target_dtype *K, target_dtype *V, target_dtype *score, target_dtype *O, int H, int S, int D) {
    target_dtype *K_T = (target_dtype*)malloc(sizeof(target_dtype) * H * S * D); //TODO: no transpose

    for (int h = 0; h < H; h++) { //for multi-head
        target_dtype *Q_h = Q + h * S * D;
        target_dtype *K_h = K + h * D * S;
        target_dtype *V_h = V + h * S * D;
        target_dtype *score_h = score + h * S * S;
        target_dtype *O_h = O + h * S * D;

        // 1. transpose K
        target_dtype *K_T_h = K_T + h * D * S;
        transpose(S, D, K_h, K_T_h);

        // 2. Q x K^T
        gemm(S, S, D, Q_h, K_T_h, score_h);

        // 3. scaling
        scaling(score_h, S, S, D);

        // 4. softmax
        softmax(S, score_h);

        // 5. Score x V
        gemm(S, D, S, score_h, V_h, O_h);
    }

    free(K_T);
}

// MHA (batched head)
void mha_batch_opt(target_dtype *Q, target_dtype *K, target_dtype *V, target_dtype *score, target_dtype *O, int H, int S, int D) {
    // scaling factor 미리 계산
    target_dtype scale = 1.0f / sqrtf((target_dtype)D);

    for (int h = 0; h < H; h++) { //for multi-head
        target_dtype *Q_h = Q + h * S * D;
        target_dtype *K_h = K + h * D * S;
        target_dtype *V_h = V + h * S * D;
        target_dtype *score_h = score + h * S * S;
        target_dtype *O_h = O + h * S * D;

        // GEMM(Q * K^T) + Scaling (kernel fusion)
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++) {
                target_dtype sum = 0.0f;
                for (int k = 0; k < D; k++) {
                    sum += Q_h[i * D + k] * K_h[j * D + k];
                }
                // 곱셈이 끝난 직후 곧바로 scale을 곱해 Scaling 루프 없앰
                score_h[i * S + j] = sum * scale;
            }
        }

        // Softmax
        softmax(S, score_h);

        // Score x V
        // (i-j-k) -> (i-k-j)
        // 안쪽 루프(j)에서 O_h와 V_h를 연속된 메모리 공간으로 순차 접근 (Cache Hit 향상)
        for (int i = 0; i < S; i++) {
            for (int k = 0; k < S; k++) {
                target_dtype s_val = score_h[i * S + k]; // 상수로 빼서 재사용
                for (int j = 0; j < D; j++) {
                    O_h[i * D + j] += s_val * V_h[k * D + j]; // O 행렬 미리 초기화했음 (calloc)
                }
            }
        }
    }
}

void mha_batch_BLAS(target_dtype *Q, target_dtype *K, target_dtype *V, target_dtype *score, target_dtype *O, int H, int S, int D) {
    target_dtype scale = 1.0f / sqrtf((target_dtype)D);

    for (int h = 0; h < H; h++) {
        target_dtype *Q_h = Q + h * S * D;
        target_dtype *K_h = K + h * D * S;
        target_dtype *V_h = V + h * S * D;
        target_dtype *score_h = score + h * S * S;
        target_dtype *O_h = O + h * S * D;

        // GEMM(Q * K^T) + Scaling (kernel fusion)
        // K_h에 대해 CblasTrans를 주어 내부적으로 전치 처리
        // alpha값에 scale을 넣어 곱셈과 동시에 스케일링 수행
        cblas_sgemm(CblasRowMajor, CblasNoTrans/*TransA*/, CblasTrans/*TransB*/,
                    S/*M*/, S/*N*/, D/*K*/,
                    scale/*alpha*/,
                    Q_h, D,
                    K_h, D,
                    0.0f/*beta*/,
                    score_h, S);

        softmax(S, score_h);

        // 3. Score * V
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    S, D, S,
                    1.0f,
                    score_h, S,
                    V_h, D,
                    0.0f,
                    O_h, D);
    }
}

void mha_batch_BLAS_batchedgemm(target_dtype *Q, target_dtype *K, target_dtype *V, target_dtype *score, target_dtype *O, int H, int S, int D) {
    target_dtype scale = 1.0f / sqrtf((target_dtype)D);

    // -------------------------------------------------------------------
    // 1. Batched GEMM을 위한 포인터 배열 세팅 (C99 VLA 사용)
    // -------------------------------------------------------------------
    const target_dtype *q_ptrs[H];
    const target_dtype *k_ptrs[H];
    const target_dtype *v_ptrs[H];
    target_dtype *score_ptrs[H];
    target_dtype *o_ptrs[H];

    for (int h = 0; h < H; h++) {
        q_ptrs[h] = Q + h * S * D;
        k_ptrs[h] = K + h * D * S;
        v_ptrs[h] = V + h * S * D;
        score_ptrs[h] = score + h * S * S;
        o_ptrs[h] = O + h * S * D;
    }
    int group_size = H;

    // GEMM 1: Q * K^T + Scaling
    CBLAS_TRANSPOSE transA1 = CblasNoTrans;
    CBLAS_TRANSPOSE transB1 = CblasTrans;
    int m1 = S, n1 = S, k1 = D;
    target_dtype alpha1 = scale, beta1 = 0.0f;
    int lda1 = D, ldb1 = D, ldc1 = S;

    cblas_sgemm_batch(CblasRowMajor,
                      &transA1, &transB1,
                      &m1, &n1, &k1,
                      &alpha1,
                      q_ptrs, &lda1,
                      k_ptrs, &ldb1,
                      &beta1,
                      score_ptrs, &ldc1,
                      1/*group count*/, &group_size/*group size*/);

    // Softmax
    for (int h = 0; h < H; h++) {
        softmax(S, score_ptrs[h]);
    }

    // GEMM 2: Score * V
    CBLAS_TRANSPOSE transA2 = CblasNoTrans;
    CBLAS_TRANSPOSE transB2 = CblasNoTrans;
    int m2 = S, n2 = D, k2 = S;
    target_dtype alpha2 = 1.0f, beta2 = 0.0f;
    int lda2 = S, ldb2 = D, ldc2 = D;

    cblas_sgemm_batch(CblasRowMajor,
                      &transA2, &transB2,
                      &m2, &n2, &k2,
                      &alpha2,
                      (const target_dtype **)score_ptrs, &lda2,
                      v_ptrs, &ldb2,
                      &beta2,
                      o_ptrs, &ldc2,
                      1, &group_size);
}

void mha_batch_BLAS_batchedgemm_opt(const target_dtype **q_ptrs, const target_dtype **k_ptrs, const target_dtype **v_ptrs, target_dtype **score_ptrs, target_dtype **o_ptrs, int total_groups/*B*H*/, int S, int D) {
    target_dtype scale = 1.0f / sqrtf((target_dtype)D);

    // GEMM 1: Q * K^T + Scaling
    CBLAS_TRANSPOSE transA1 = CblasNoTrans;
    CBLAS_TRANSPOSE transB1 = CblasTrans;
    int m1 = S, n1 = S, k1 = D;
    target_dtype alpha1 = scale, beta1 = 0.0f;
    int lda1 = D, ldb1 = D, ldc1 = S;

    cblas_sgemm_batch(CblasRowMajor,
                      &transA1, &transB1,
                      &m1, &n1, &k1,
                      &alpha1,
                      q_ptrs, &lda1,
                      k_ptrs, &ldb1,
                      &beta1,
                      score_ptrs, &ldc1,
                      1, &total_groups); // 그룹 크기가 B * H 인 단일 그룹

    // Softmax: 포인터 배열을 순회하며 처리
    for (int i = 0; i < total_groups; i++) {
        softmax(S, score_ptrs[i]);
    }

    // GEMM 2: Score * V
    CBLAS_TRANSPOSE transA2 = CblasNoTrans;
    CBLAS_TRANSPOSE transB2 = CblasNoTrans;
    int m2 = S, n2 = D, k2 = S;
    target_dtype alpha2 = 1.0f, beta2 = 0.0f;
    int lda2 = S, ldb2 = D, ldc2 = D;

    cblas_sgemm_batch(CblasRowMajor,
                      &transA2, &transB2,
                      &m2, &n2, &k2,
                      &alpha2,
                      (const target_dtype **)score_ptrs, &lda2,
                      v_ptrs, &ldb2,
                      &beta2,
                      o_ptrs, &ldc2,
                      1, &total_groups);
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    int B = 4;      // Number of Batches
    int H = 16;     // Number of Heads
    int S = 128;     // Sequence Length per Head
    int D = 256;     // Head dimensions (length of vector)
    double start, end;

    if (argc >= 5) {
        B = atoi(argv[1]);
        H = atoi(argv[2]);
        S = atoi(argv[3]);
        D = atoi(argv[4]);
        printf("[Config] Batch: %d, Head: %d, SeqLen: %d, HeadDim: %d\n", B, H, S, D);
    } else {
        printf("Usage: %s <B(#batch)> <H(#head)> <S(seq_len)> <D(head_dim)>\n", argv[0]);
        printf("[Config(default)] Batch: %d, Head: %d, SeqLen: %d, HeadDim: %d\n", B, H, S, D);
    }
    int total = B * H * S * D;

    //Init
    target_dtype* Q = (target_dtype*)malloc(total * sizeof(target_dtype));
    target_dtype* K = (target_dtype*)malloc(total * sizeof(target_dtype));
    target_dtype* V = (target_dtype*)malloc(total * sizeof(target_dtype));
    target_dtype *score = (target_dtype*)calloc(B * H * S * S, sizeof(target_dtype));
    target_dtype* O1 = (target_dtype*)calloc(total, sizeof(target_dtype));    // Score * V
    target_dtype* O2 = (target_dtype*)calloc(total, sizeof(target_dtype));    // Score * V
    target_dtype* O3 = (target_dtype*)calloc(total, sizeof(target_dtype));    // Score * V
    target_dtype* O4 = (target_dtype*)calloc(total, sizeof(target_dtype));    // Score * V
    target_dtype* O5 = (target_dtype*)calloc(total, sizeof(target_dtype));    // Score * V

    init_matrix(Q, total);
    init_matrix(K, total);
    init_matrix(V, total);

    //1. mha_batch (default - naive gemm)
    start = now_sec();
    for (int b = 0; b < B; b++) {
        target_dtype *q = Q + b * H * S * D;
        target_dtype *k = K + b * H * S * D;
        target_dtype *v = V + b * H * S * D;
        target_dtype *s = score + b * H * S * S;
        target_dtype *o = O1 + b * H * S * D;
        mha_batch(q, k, v, s, o, H, S, D);
    }
    end = now_sec();
    printf("(Base) Elapsed time: %f sec\n", end - start);

    //2. mha_batch_opt (naive gemm but a little optimization)
    start = now_sec();
    for (int b = 0; b < B; b++) {
        target_dtype *q = Q + b * H * S * D;
        target_dtype *k = K + b * H * S * D;
        target_dtype *v = V + b * H * S * D;
        target_dtype *s = score + b * H * S * S;
        target_dtype *o = O2 + b * H * S * D;
        mha_batch_opt(q, k, v, s, o, H, S, D);
    }
    end = now_sec();
    printf("(Opt) Elapsed time: %f sec\n", end - start);

    //3. using BLAS lib.
    start = now_sec();
    for (int b = 0; b < B; b++) {
        target_dtype *q = Q + b * H * S * D;
        target_dtype *k = K + b * H * S * D;
        target_dtype *v = V + b * H * S * D;
        target_dtype *s = score + b * H * S * S;
        target_dtype *o = O3 + b * H * S * D;
        mha_batch_BLAS(q, k, v, s, o, H, S, D);
    }
    end = now_sec();
    printf("(BLAS) Elapsed time: %f sec\n", end - start);

    //4. using BLAS lib + batched gemm (at head-level)
    start = now_sec();
    for (int b = 0; b < B; b++) {
        target_dtype *q = Q + b * H * S * D;
        target_dtype *k = K + b * H * S * D;
        target_dtype *v = V + b * H * S * D;
        target_dtype *s = score + b * H * S * S;
        target_dtype *o = O4 + b * H * S * D;
        mha_batch_BLAS_batchedgemm(q, k, v, s, o, H, S, D);
    }
    end = now_sec();
    printf("(BLAS_batchedgemm) Elapsed time: %f sec\n", end - start);

    //5. using BLAS lib + batched gemm (at batch-level)
    int total_groups = B * H; // 전체 처리해야 할 행렬(Head)의 총합
    const target_dtype **q_ptrs = (const target_dtype **)malloc(total_groups * sizeof(target_dtype*));
    const target_dtype **k_ptrs = (const target_dtype **)malloc(total_groups * sizeof(target_dtype*));
    const target_dtype **v_ptrs = (const target_dtype **)malloc(total_groups * sizeof(target_dtype*));
    target_dtype **score_ptrs = (target_dtype **)malloc(total_groups * sizeof(target_dtype*));
    target_dtype **o_ptrs = (target_dtype **)malloc(total_groups * sizeof(target_dtype*));
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int idx = b * H + h; // 1D idx
            q_ptrs[idx] = Q + idx * S * D;
            k_ptrs[idx] = K + idx * S * D;
            v_ptrs[idx] = V + idx * S * D;
            score_ptrs[idx] = score + idx * S * S;
            o_ptrs[idx] = O5 + idx * S * D;
        }
    }
    start = now_sec();
    mha_batch_BLAS_batchedgemm_opt(q_ptrs, k_ptrs, v_ptrs, score_ptrs, o_ptrs, total_groups, S, D);
    end = now_sec();
    printf("(BLAS_batchedgemm_opt) Elapsed time: %f sec\n", end - start);

    //Verify
    verify_matrix(B * H * S * D, O1, O2, "mha_batch_opt");
    verify_matrix(B * H * S * D, O1, O3, "mha_batch_BLAS");
    verify_matrix(B * H * S * D, O1, O4, "mha_batch_BLAS_batchedgemm");
    verify_matrix(B * H * S * D, O1, O5, "mha_batch_BLAS_batchedgemm_opt");
    printf("Attention Computation Completed.\n");

    free(Q); free(K); free(V); free(score); free(O1); free(O2); free(O3); free(O4); free(O5);
    return 0;
}