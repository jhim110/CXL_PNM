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
        A[i] = ((target_dtype)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

void transpose(int M, int N, target_dtype *A, target_dtype *B) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[j*M + i] = A[i*N + j];
        }
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

// 1D Softmax
void softmax_1d(int S_total, target_dtype *score) {
    target_dtype max_val = score[0];
    for (int s = 1; s < S_total; s++) {
        if (score[s] > max_val) max_val = score[s];
    }
    
    target_dtype sum = 0;
    for (int s = 0; s < S_total; s++) {
        score[s] = expf(score[s] - max_val); // float 최적화를 위해 expf 사용
        sum += score[s];
    }
    
    for (int s = 0; s < S_total; s++) {
        score[s] /= sum;
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

// y = alpha * A * x + beta * y 의 단순화 버전
// trans_A == 0: y = A * x   (A: M x N 행렬, x: N 길이 벡터, y: M 길이 벡터)
// trans_A == 1: y = A^T * x (A: M x N 행렬, x: M 길이 벡터, y: N 길이 벡터)
void gemv(int M, int N, target_dtype *A, target_dtype *x, target_dtype *y, int trans_A) {
    if (trans_A == 0) {
        // y = A * x
        for (int i = 0; i < M; i++) {
            target_dtype sum = 0;
            target_dtype *A_row = A + i * N;
            for (int j = 0; j < N; j++) {
                sum += A_row[j] * x[j];
            }
            y[i] = sum;
        }
    } else {
        // y = A^T * x
        for (int j = 0; j < N; j++) {
            y[j] = 0;
        }
        for (int i = 0; i < M; i++) {
            target_dtype x_val = x[i];
            target_dtype *A_row = A + i * N;
            for (int j = 0; j < N; j++) {
                y[j] += x_val * A_row[j];
            }
        }
    }
}

void update_kv_cache(target_dtype *k_current, target_dtype *v_current, 
                     target_dtype *K_cache, target_dtype *V_cache, 
                     int B, int H, int D, int max_seq_len, int current_pos) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            target_dtype *k_cache_dest = K_cache + b * H * max_seq_len * D + h * max_seq_len * D + current_pos * D;
            target_dtype *v_cache_dest = V_cache + b * H * max_seq_len * D + h * max_seq_len * D + current_pos * D;
            
            target_dtype *k_src = k_current + b * H * D + h * D;
            target_dtype *v_src = v_current + b * H * D + h * D;

            for (int d = 0; d < D; d++) {
                k_cache_dest[d] = k_src[d];
                v_cache_dest[d] = v_src[d];
            }
        }
    }
}

void linear_projection(target_dtype *out, int size) {
    // TODO: gemv(D, D, x, W, out) 교체 필요
    // out = x * W
    // verify matrix 수행 위해, linear projection 단계에서 매번 동일한 값을 주도록 수정. 
    for (int i = 0; i < size; i++) {
        out[i] = (target_dtype)(i % 100) * 0.01f; // 0.0 ~ 0.99 사이의 순차적인 값
    }
}

// MHA 결과와 MQA 검증을 위해 K, V의 모든 헤드 값을 0번째 헤드와 동일하게 강제 복사하도록 함.
void linear_projection_shared_kv(target_dtype *out, int B, int H, int D) {
    for (int b = 0; b < B; b++) {
        // 현재 배치(b)의 0번째 헤드 포인터
        target_dtype* head_0 = out + b * H * D; 
        
        // 1. 0번째 헤드에 임의의 값 할당 (디버깅을 위해 순차적인 값 사용)
        for (int d = 0; d < D; d++) {
            head_0[d] = (target_dtype)((b * D + d) % 100) * 0.01f;
        }

        // 2. 1번째 헤드부터 H-1번째 헤드까지 0번째 헤드의 값을 그대로 복사!
        for (int h = 1; h < H; h++) {
            target_dtype* head_h = out + b * H * D + h * D;
            for (int d = 0; d < D; d++) {
                head_h[d] = head_0[d]; // 0번째 헤드 값 덮어쓰기
            }
        }
    }
}

typedef void (*attention_func_t)(target_dtype *Q, target_dtype *K_cache, target_dtype *V_cache, 
    target_dtype *score, target_dtype *O, 
    int H, int S_total, int max_seq_len, int D);

void mha_base(target_dtype *Q, target_dtype *K_cache, target_dtype *V_cache, 
              target_dtype *score, target_dtype *O, 
              int H, int S_total, int max_seq_len, int D) {
    for (int h = 0; h < H; h++) {
        target_dtype *q_h = Q + h * D; 
        target_dtype *k_cache_h = K_cache + h * max_seq_len * D; 
        target_dtype *v_cache_h = V_cache + h * max_seq_len * D;
        target_dtype *score_h = score + h * max_seq_len; 
        target_dtype *o_h = O + h * D;

        // Q x K^T
        // Out(=score_h) = q_h * K_cache_h 수행
        // Vec_Q(D x 1), Mat_K(S_total x D), Out(S_total x 1)
        // S_total=M, D=N, Mat_k는 notrans.
        //gemv(M, N, Mat_A, Vec_x, Out_y, trans_A)
        gemv(S_total, D, k_cache_h, q_h, score_h, 0);

        //scaling
        for (int s = 0; s < S_total; s++) {
            score_h[s] /= sqrtf((float)D);
        }

        //Softmax
        softmax_1d(S_total, score_h); 

        // Score x V
        // Vec_Score(S_total x 1), Mat_V(S_total x D), Out(D x 1)
        // M = S_total, N = D, Mat_V는 transpose 해야함.
        //gemv(M, N, Mat_A, Vec_x, Out_y, trans_A)
        gemv(S_total, D, v_cache_h, score_h, o_h, 1);
    }
}

void mha_openblas(target_dtype *Q, target_dtype *K_cache, target_dtype *V_cache, 
                  target_dtype *score, target_dtype *O,
                  int H, int S_total, int max_seq_len, int D) {
// 1. BLAS 연산을 위한 상수 미리 계산
    float alpha_q_k = 1.0f / sqrtf((float)D); // Q x K^T 의 스케일링을 alpha에 통합!
    float alpha_v   = 1.0f;
    float beta      = 0.0f;                   // 기존 배열을 0으로 덮어쓰기 위해 beta는 0

    for (int h = 0; h < H; h++) {
        target_dtype *q_h = Q + h * D; 
        target_dtype *k_cache_h = K_cache + h * max_seq_len * D; 
        target_dtype *v_cache_h = V_cache + h * max_seq_len * D;
        target_dtype *score_h = score + h * max_seq_len; 
        target_dtype *o_h = O + h * D;

        // =================================================================
        // 1. Q x K^T 및 Scaling 동시 수행
        // 수식: score_h = (1/sqrt(D)) * k_cache_h * q_h + (0 * score_h)
        // =================================================================
        cblas_sgemv(CblasRowMajor, CblasNoTrans, 
                    S_total, D,           // M, N
                    alpha_q_k,            // alpha (스케일링 통합)
                    k_cache_h, D,         // A 행렬과 lda(Row Stride)
                    q_h, 1,               // x 벡터와 incX
                    beta,                 // beta
                    score_h, 1);          // y 벡터와 incY
        // (기존의 scaling for 루프 삭제)

        // =================================================================
        // 2. Softmax
        // =================================================================
        softmax_1d(S_total, score_h); 

        // =================================================================
        // 3. Score x V 수행
        // 수식: o_h = 1.0 * v_cache_h^T * score_h + (0 * o_h)
        // =================================================================
        cblas_sgemv(CblasRowMajor, CblasTrans, // V는 Transpose 수행 (trans_A = 1과 동일)
                    S_total, D,           // M, N (TransA를 써도 원래 M, N을 넣는 것이 BLAS 규칙)
                    alpha_v,              // alpha
                    v_cache_h, D,         // A 행렬과 lda
                    score_h, 1,           // x 벡터와 incX
                    beta,                 // beta
                    o_h, 1);              // y 벡터와 incY
    }
}

void mqa_openblas(target_dtype *Q, target_dtype *K_cache, target_dtype *V_cache, 
                  target_dtype *score, target_dtype *O,
                  int H, int S_total, int max_seq_len, int D) {
                  
    float alpha_q_k = 1.0f / sqrtf((float)D);
    float alpha_v   = 1.0f;
    float beta      = 0.0f;

    // MQA는 모든 헤드가 동일한 K, V를 공유
    // mha처럼 h 루프를 돌며 k_cache_h 오프셋을 구하지 않고, 
    // 무조건 첫 번째 헤드(h=0)의 캐시 포인터를 전체 연산에 재사용.
    target_dtype *k_cache_shared = K_cache; 
    target_dtype *v_cache_shared = V_cache; 

    // =================================================================
    // 1. Q x K^T (GEMM을 통해 H loop 제거)
    // Q는 [H, D] 크기의 행렬, K는 [S_total, D] 크기의 행렬
    // 결과값 score는 [H, max_seq_len] 공간 내의 [H, S_total] 부분
    // =================================================================
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                H, S_total, D,          // M(Q의 행=H), N(K의 행=S_total), K_dim(D)
                alpha_q_k, 
                Q, D,                   // A 행렬(Q), lda(Row Stride) = D
                k_cache_shared, D,      // B 행렬(K), ldb = D
                beta, 
                score, max_seq_len);    // C 행렬(Score), ldc = max_seq_len (배열의 실제 가로 길이)

    // =================================================================
    // 2. Softmax (각 행별 연산이 필요하므로 이것만 루프 유지)
    // =================================================================
    for (int h = 0; h < H; h++) {
        target_dtype *score_h = score + h * max_seq_len;
        softmax_1d(S_total, score_h); 
    }

    // =================================================================
    // 3. Score x V
    // Score는 [H, S_total], V는 [S_total, D] 크기의 행렬
    // 결과값 O는 [H, D] 크기의 행렬
    // =================================================================
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, // 이번엔 V를 Transpose 하지 않음
                H, D, S_total,          // M(Score의 행=H), N(V의 열=D), K_dim(S_total)
                alpha_v, 
                score, max_seq_len,     // A 행렬(Score), lda = max_seq_len
                v_cache_shared, D,      // B 행렬(V), ldb = D
                beta, 
                O, D);                  // C 행렬(O), ldc = D
}

void decode_batch(int gen_tokens, int prompt_len, int max_seq_len, 
                     int B, int H, int D, int q_size,
                     target_dtype* Q, target_dtype* k_current, target_dtype* v_current,
                     target_dtype* K_cache, target_dtype* V_cache, target_dtype* score, target_dtype* O,
                     attention_func_t mha) {
    // Decoding - Attention (FFN 없이 Attention 과정 통과하면 Token 추가된다고 가정)
    for (int step = 0; step < gen_tokens; step++) {
        // Current S = Prompt Length + 생성 Token 수 (Step) + 1
        int S_total = prompt_len + step + 1; 
        
        if (S_total > max_seq_len) {
            printf("Error: Exceeded maximum sequence length (KV Cache full)!\n");
            break;
        }

        // 새롭게 생성할 토큰의 Query 행렬 생성
        // x: input vector (WTE+WPE, 이전 layer에서 넘어온 현재 토큰의 입력 벡터 (1 X D))
        // Q = gemv(D, D, x, W_q, Q);
        linear_projection(Q, q_size);         // Q = x * W_q

        // KV Cache Update
        // k_current = x * W_k (gemv(D, D, x, W_k, K_current))
        // v_current = x * W_v (gemv(D, D, x, W_v, v_current))
        // MQA 검증을 위해 별도의 linear_projection 함수 사용
          //linear_projection(k_current, q_size); // K = x * W_k
          //linear_projection(v_current, q_size); // V = x * W_v
        linear_projection_shared_kv(k_current, B, H, D); 
        linear_projection_shared_kv(v_current, B, H, D);
        int current_pos = S_total - 1;
        update_kv_cache(k_current, v_current, K_cache, V_cache, B, H, D, max_seq_len, current_pos);

        // 배치 별 Attention 수행
        double step_start = now_sec();
        for (int b = 0; b < B; b++) {
            target_dtype *q = Q + b * H * 1 * D;
            target_dtype *k_c = K_cache + b * H * max_seq_len * D;
            target_dtype *v_c = V_cache + b * H * max_seq_len * D;
            target_dtype *s = score + b * H * max_seq_len;
            target_dtype *o = O + b * H * 1 * D;

            mha(q, k_c, v_c, s, o, H, S_total, max_seq_len, D);
        }
        double step_end = now_sec();

        //printf("  - Token %2d generated (S_total: %4d) | Elapsed time: %f sec\n", step + 1, S_total, step_end - step_start);
    }
}

int main(int argc, char *argv[]) {
    //srand((unsigned int)time(NULL));
    srand(1234);

    int B = 1;      // Number of Batches
    int H = 16;     // Number of Heads
    int D = 256;     // Head dimensions (vector length)
    int max_seq_len = 2048; // KV Cache로 미리 할당해둘 최대 토큰 길이 (물리적 메모리 한계)
    int prompt_len = 128;   // Prefill 단계에서 이미 처리되어 캐시에 들어있는 토큰 수
    int gen_tokens = 10;    // 디코딩 단계에서 새롭게 생성할 토큰의 수
    double start, end;

    if (argc >= 6) {
        B = atoi(argv[1]);
        H = atoi(argv[2]);
        D = atoi(argv[3]);
        prompt_len = atoi(argv[4]);
        gen_tokens = atoi(argv[5]);
    } else {
        printf("Using default values. (User did not put arg or put wrong values)\n");
    }
    printf("[Config] Batch: %d, Head: %d, HeadDim: %d\n", B, H, D);
    printf("[Status] KV Cache Max: %d, Prompt length: %d, Generating %d tokens...\n", max_seq_len, prompt_len, gen_tokens);

    // 메모리 할당 (KV Cache는 전체 max_seq_len 만큼 미리 할당)
    int q_size = B * H * 1 * D;
    int cache_size = B * H * max_seq_len * D;
    int score_size = B * H * max_seq_len;
    
    target_dtype* K_cache = (target_dtype*)malloc(cache_size * sizeof(target_dtype));
    target_dtype* V_cache = (target_dtype*)malloc(cache_size * sizeof(target_dtype));
    target_dtype* score = (target_dtype*)calloc(score_size, sizeof(target_dtype));
    target_dtype* Q = (target_dtype*)malloc(q_size * sizeof(target_dtype));
    target_dtype* O1 = (target_dtype*)calloc(q_size, sizeof(target_dtype)); 
    target_dtype* O2 = (target_dtype*)calloc(q_size, sizeof(target_dtype)); 
    target_dtype* O3 = (target_dtype*)calloc(q_size, sizeof(target_dtype)); 
    
    // 현재 토큰의 K, V [B, H, 1, D]
    target_dtype* k_current = (target_dtype*)malloc(q_size * sizeof(target_dtype));
    target_dtype* v_current = (target_dtype*)malloc(q_size * sizeof(target_dtype));

    // KV Cache Init (Prefill에서 넘어온 prompt_len 만큼의 과거 K, V가 이미 있다고 가정)
    init_matrix(K_cache, cache_size); 
    init_matrix(V_cache, cache_size);

    // 1. decode_batch with mha_base (native gemv 사용)
    start = now_sec();
    decode_batch(gen_tokens, prompt_len, max_seq_len, B, H, D, q_size, Q, k_current, v_current, K_cache, V_cache, score, O1, mha_base);
    end = now_sec();
    printf("(decode_batch with %s) Total elapsed time for %d tokens: %f sec\n", "mha_base", gen_tokens, end - start);

    // 2. decode_batch with mha_openblas (openblas gemv library 사용)
    start = now_sec();
    decode_batch(gen_tokens, prompt_len, max_seq_len, B, H, D, q_size, Q, k_current, v_current, K_cache, V_cache, score, O2, mha_openblas);
    end = now_sec();
    printf("(decode_batch with %s) Total elapsed time for %d tokens: %f sec\n", "mha_openblas", gen_tokens, end - start);

    // 3. decode_batch with mqa_openblas (openblas gemm library 사용)
    start = now_sec();
    decode_batch(gen_tokens, prompt_len, max_seq_len, B, H, D, q_size, Q, k_current, v_current, K_cache, V_cache, score, O3, mqa_openblas);
    end = now_sec();
    printf("(decode_batch with %s) Total elapsed time for %d tokens: %f sec\n", "mqa_openblas", gen_tokens, end - start);

    //Verify
    verify_matrix(q_size, O1, O2, "decode_batch(mha_openblas)");
    verify_matrix(q_size, O1, O3, "decode_batch(mqa_openblas)");

    free(Q); free(K_cache); free(V_cache); free(score);
    free(O1); free(O2); free(O3);
    free(k_current); free(v_current);

    return 0;
}