#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 데이터 타입 정의
typedef float target_dtype;

void init_matrix(target_dtype *A, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        A[i] = (target_dtype)((float)rand() / RAND_MAX);
    }
}

//Naive
void gemm_qk(target_dtype* Q, target_dtype* K, target_dtype* S, int M, int N, int D) { //S = Q x K^T
    // GEMM1. Q * K^T → [M x D] x [N x D] → [M x N]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            target_dtype sum = 0.0f;
            for (int k = 0; k < D; k++) {
                sum += Q[i*D + k] * K[j*D + k];
            }
            S[i*N + j] = sum;
        }
    }
}

//Naive
void gemm(target_dtype* A, target_dtype* B, target_dtype* C, int M, int K, int N) { //C = A x B
    // GEMM2. Score * V → [M x N] x [N x D] → [M x D]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            target_dtype sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j]; //TODO: col-major to row-major
            }
            C[i*N + j] = sum;
        }
    }
}

void scaling(target_dtype* S, int M, int N, int D) {
    target_dtype scale = 1.0f / sqrtf((target_dtype)D);
    for (int i = 0; i < M * N; i++) {
        S[i] *= scale;
    }
}

void softmax(target_dtype *S, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        target_dtype max_val = S[i*cols];

        // find max (numerical stability)
        for (int j = 1; j < cols; j++) {
            if (S[i*cols + j] > max_val)
                max_val = S[i*cols + j];
        }

        target_dtype sum = 0.0f;

        // exp + accumulate
        for (int j = 0; j < cols; j++) {
            S[i*cols + j] = expf((float)S[i*cols + j] - max_val);
            sum += S[i*cols + j];
        }

        // normalize
        for (int j = 0; j < cols; j++) {
            S[i*cols + j] /= sum;
        }
    }
}

void attention_MHA(target_dtype* Q, target_dtype* K, target_dtype* V, target_dtype* S, target_dtype* O, int M, int N, int D) {
    // GEMM1. S = Q * K^T = [M x D] x [N x D] = [M x N]
    gemm_qk(Q, K, S, M, N, D);

    // Scaling and Softmax(S)
    scaling(S, M, N, D);
    softmax(S, M, N);

    // GEMM2. O = S * V = [M x N] x [N x D] = [M x D]
    gemm(S, V, O, M, N, D);
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    //Defaulty
    int M = 32;     // Number of Qeury Row (#Batch * #Head), Reshaping  [B, H, Tq, D] to [B*H, Tq, D]
    int N = 128;    // KV Cache Length
    int D = 64;     // (head) dimensions (length of vector)

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        D = atoi(argv[3]);
        printf("[Config] M: %d, N: %d, D: %d\n", M, N, D);
    } else {
        printf("Usage: %s <M(Batch*Head)> <N(KV_Len)> <D(Head_Dim)>\n", argv[0]);
        printf("[Config(default)]: M: %d, N: %d, D: %d\n\n", M, N, D);
    }

    target_dtype* Q = (target_dtype*)malloc(M * D * sizeof(target_dtype));
    target_dtype* K = (target_dtype*)malloc(N * D * sizeof(target_dtype));
    target_dtype* V = (target_dtype*)malloc(N * D * sizeof(target_dtype));
    target_dtype* Score = (target_dtype*)malloc(M * N * sizeof(target_dtype));  // Q * K^T
    target_dtype* Out = (target_dtype*)malloc(M * D * sizeof(target_dtype));    // Score * V

    init_matrix(Q, M, D);
    init_matrix(K, N, D); //TODO: Sinlge KV cache to multi
    init_matrix(V, N, D);

    attention_MHA(Q, K, V, Score, Out, M, N, D);

    printf("Attention Computation Completed.\n");
    printf("Sample Output[0]: %f\n", (float)Out[0]);

    free(Q); free(K); free(V); free(Score); free(Out);
    return 0;
}