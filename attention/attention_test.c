#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 1. 데이터 타입 정의: 이 부분을 _Float16 등으로 변경하세요.
typedef float target_dtype;

// 2. 수학 함수 매크로: 데이터 타입 변경 시 (예: _Float16의 경우 형변환 혹은 다른 함수로) 맞게 수정하세요.
#define MATH_EXP(x) expf(x)
#define MATH_SQRT(x) sqrtf(x)

// Q * K^T 연산
void matmul_q_kt(int M, int N, int K, const target_dtype* Q, const target_dtype* K_mat, target_dtype* Out) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            target_dtype sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += Q[i * K + k] * K_mat[j * K + k];
            }
            Out[i * N + j] = sum;
        }
    }
}

// 스케일링 및 Softmax 연산
void scale_softmax(int M, int N, target_dtype scale, target_dtype* Score) {
    for (int i = 0; i < M; i++) {
        target_dtype max_val = -INFINITY;
        
        // Max 값 찾기
        for (int j = 0; j < N; j++) {
            target_dtype val = Score[i * N + j] * scale;
            Score[i * N + j] = val;
            if (val > max_val) max_val = val;
        }

        // Exp 계산 및 합계 구하기
        target_dtype sum_exp = 0.0f;
        for (int j = 0; j < N; j++) {
            Score[i * N + j] = (target_dtype)MATH_EXP((float)(Score[i * N + j] - max_val));
            sum_exp += Score[i * N + j];
        }

        // 정규화 (Softmax)
        for (int j = 0; j < N; j++) {
            Score[i * N + j] /= sum_exp;
        }
    }
}

// Score * V 연산
void matmul_score_v(int M, int N, int K, const target_dtype* Score, const target_dtype* V, target_dtype* Out) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            target_dtype sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += Score[i * K + k] * V[k * N + j];
            }
            Out[i * N + j] = sum;
        }
    }
}

int main() {
    srand((unsigned int)time(NULL));

    int seq_len = 128;
    int head_dim = 64;

    target_dtype* Q = (target_dtype*)malloc(seq_len * head_dim * sizeof(target_dtype));
    target_dtype* K_mat = (target_dtype*)malloc(seq_len * head_dim * sizeof(target_dtype));
    target_dtype* V = (target_dtype*)malloc(seq_len * head_dim * sizeof(target_dtype));
    target_dtype* Score = (target_dtype*)malloc(seq_len * seq_len * sizeof(target_dtype));
    target_dtype* Out = (target_dtype*)malloc(seq_len * head_dim * sizeof(target_dtype));

    // 데이터 초기화 (타입 변경 시 경고를 방지하기 위해 float에서 캐스팅)
    for(int i = 0; i < seq_len * head_dim; i++) {
        Q[i] = (target_dtype)((float)rand() / RAND_MAX);
        K_mat[i] = (target_dtype)((float)rand() / RAND_MAX);
        V[i] = (target_dtype)((float)rand() / RAND_MAX);
    }

    matmul_q_kt(seq_len, seq_len, head_dim, Q, K_mat, Score);

    target_dtype scale = (target_dtype)(1.0f / MATH_SQRT((float)head_dim));
    scale_softmax(seq_len, seq_len, scale, Score);

    matmul_score_v(seq_len, head_dim, seq_len, Score, V, Out);

    printf("Attention Computation Completed.\n");
    // 출력 시 포맷 지정자(%f)와의 호환성을 위해 float으로 명시적 형변환
    printf("Sample Output[0]: %f\n", (float)Out[0]);

    free(Q); free(K_mat); free(V); free(Score); free(Out);
    return 0;
}