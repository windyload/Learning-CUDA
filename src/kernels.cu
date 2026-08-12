#include <cstddef>
#include <cstdio>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <stdexcept>
#include <vector>

#include "../tester/utils.h"

// 为不同的数据类型 T 定义对应的向量化处理规则。float 使用 float4，每次处理 4
// 个元素；half 使用 half2，每次处理 2 个元素。VecTraits
// 统一封装向量类型、向量宽度、向量化 load/store、平方和计算以及 RMSNorm
// 归一化操作，使后续 kernel 可以使用统一接口处理不同数据类型。
template <typename T> struct VecTraits;

template <> struct VecTraits<float> {
  using Vec = float4;

  static constexpr int WIDTH = 4;

  __device__ __forceinline__ static Vec load(const float *ptr, int vec_idx) {
    return reinterpret_cast<const float4 *>(ptr)[vec_idx];
  }

  __device__ __forceinline__ static void store(float *ptr, int vec_idx, Vec v) {
    reinterpret_cast<float4 *>(ptr)[vec_idx] = v;
  }

  __device__ __forceinline__ static float sumSq(Vec v) {
    return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }

  __device__ __forceinline__ static Vec normalize(Vec x, Vec w, float inv_rms) {
    Vec y;

    y.x = x.x * inv_rms * w.x;
    y.y = x.y * inv_rms * w.y;
    y.z = x.z * inv_rms * w.z;
    y.w = x.w * inv_rms * w.w;

    return y;
  }
};

template <> struct VecTraits<half> {
  using Vec = half2;

  static constexpr int WIDTH = 2;

  __device__ __forceinline__ static Vec load(const half *ptr, int vec_idx) {
    return reinterpret_cast<const half2 *>(ptr)[vec_idx];
  }

  __device__ __forceinline__ static void store(half *ptr, int vec_idx, Vec v) {
    reinterpret_cast<half2 *>(ptr)[vec_idx] = v;
  }

  __device__ __forceinline__ static float sumSq(Vec v) {
    float2 f = __half22float2(v);

    return f.x * f.x + f.y * f.y;
  }

  __device__ __forceinline__ static Vec normalize(Vec x, Vec w, float inv_rms) {
    float2 xf = __half22float2(x);
    float2 wf = __half22float2(w);

    return __floats2half2_rn(xf.x * inv_rms * wf.x, xf.y * inv_rms * wf.y);
  }
};

// 测试发现 hidden_dim
// 可能无法满足向量化访问的对齐/整除要求，此时需要退化为标量访问。为了统一 float
// 和 half 在标量路径中的 FP32 计算逻辑，封装 ScalarOps，负责标量类型与 float
// 之间的转换。
template <typename T> struct ScalarOps;

template <> struct ScalarOps<float> {
  __device__ __forceinline__ static float toFloat(float x) { return x; }

  __device__ __forceinline__ static float fromFloat(float x) { return x; }
};

template <> struct ScalarOps<half> {
  __device__ __forceinline__ static float toFloat(half x) {
    return __half2float(x);
  }

  __device__ __forceinline__ static half fromFloat(float x) {
    return __float2half_rn(x);
  }
};

// 使用线程束洗牌函数进行线程束规约求和
__device__ __forceinline__ float warpReduceSum(float val) {

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }

  return val;
}

// 若hidden_dim满足向量化访问的对齐/整除要求，使用此函数
template <typename T, int HIDDEN_DIM, int BLOCK_SIZE>
__global__ void
rmsNormVectorKernelV5(const T *__restrict__ input, const T *__restrict__ weight,
                      T *__restrict__ output, size_t rows, float eps) {
  using Traits = VecTraits<T>;

  // 每线程负责多少个float4/half2向量，这决定后面的循环次数
  constexpr int VEC_SIZE = Traits::WIDTH;
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  constexpr int NUM_VECS = HIDDEN_DIM / VEC_SIZE;
  constexpr int VECS_PER_THREAD = (NUM_VECS + BLOCK_SIZE - 1) / BLOCK_SIZE;

  static_assert(HIDDEN_DIM % VEC_SIZE == 0,
                "HIDDEN_DIM must be divisible by vector width");

  // 线程id，线程在线程束中的id，以及线程所属线程束id
  const int tid = threadIdx.x;
  const int lane_id = tid & 31;
  const int warp_id = tid / 32;

  // 定义共享内存
  // 每个warp需要存储一个partial sum
  __shared__ float warp_sums[NUM_WARPS];
  // 整个row(一个block)需要共用一个inv_rms，通过shared memory进行block内广播
  __shared__ float s_inv_rms;

  // 当前row在input中的起始位置,在output中的起始位置
  const size_t row_offset = blockIdx.x * HIDDEN_DIM;
  const T *row_input = input + row_offset;
  T *row_output = output + row_offset;

  // 1. 数据从Global -> Register，计算local sum(x^2)
  using Vec = typename Traits::Vec;
  // x 和 x^2 都保存在寄存器中，用于后续计算
  Vec x_reg[VECS_PER_THREAD];
  float local_sum_sq = 0.0f;

#pragma unroll
  for (int i = 0; i < VECS_PER_THREAD; ++i) {
    const int vec_idx = tid + i * BLOCK_SIZE;

    if (vec_idx < NUM_VECS) {
      Vec x = Traits::load(row_input, vec_idx);

      x_reg[i] = x;

      // 为保证精度，无论是float还是half，均使用FP32累加
      local_sum_sq += Traits::sumSq(x);
    }
  }

  // 2. 对于block内的每一个warp，均使用线程束洗牌函数进行线程束内规约，32->1
  float warp_sum = warpReduceSum(local_sum_sq);
  // 将线程束规约结果写入线程束的第一个线程中
  if (lane_id == 0) {
    warp_sums[warp_id] = warp_sum;
  }
  __syncthreads();

  // 3.将block内NUM_WARPS个线程束规约结果，再次进行规约，最终得到row的平方和，据此计算inv_rms
  if (warp_id == 0) {
    float block_sum = lane_id < NUM_WARPS ? warp_sums[lane_id] : 0.0f;

    block_sum = warpReduceSum(block_sum);

    // lane 0 得到整个 row 的 sum(x^2)
    if (lane_id == 0) {
      s_inv_rms = rsqrtf(block_sum / static_cast<float>(HIDDEN_DIM) + eps);
    }
  }
  __syncthreads();

  const float inv_rms = s_inv_rms;

// 4. 计算x_i * inv_rms * weight_i得到RMSNorm
#pragma unroll
  for (int i = 0; i < VECS_PER_THREAD; ++i) {
    const int vec_idx = tid + i * BLOCK_SIZE;

    if (vec_idx < NUM_VECS) {
      const Vec x = x_reg[i];
      const Vec w = Traits::load(weight, vec_idx);
      const Vec y = Traits::normalize(x, w, inv_rms);

      Traits::store(row_output, vec_idx, y);
    }
  }
}

// 用于：hidden_dim = 1, hidden_dim = 31, hidden_dim = 769
// 以及其它无法安全向量化的尺寸。
template <typename T, int HIDDEN_DIM, int BLOCK_SIZE>
__global__ void
rmsNormScalarKernelV5(const T *__restrict__ input, const T *__restrict__ weight,
                      T *__restrict__ output, size_t rows, float eps) {

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  constexpr int ITEMS_PER_THREAD = (HIDDEN_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const int tid = threadIdx.x;
  const int lane_id = tid & 31;
  const int warp_id = tid >> 5;

  __shared__ float warp_sums[NUM_WARPS];
  __shared__ float s_inv_rms;

  const size_t row_offset = blockIdx.x * HIDDEN_DIM;

  T x_reg[ITEMS_PER_THREAD];
  float local_sum_sq = 0.0f;

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int col = tid + i * BLOCK_SIZE;

    if (col < HIDDEN_DIM) {
      const T x_raw = input[row_offset + col];
      x_reg[i] = x_raw;
      const float x = ScalarOps<T>::toFloat(x_raw);

      local_sum_sq += x * x;
    }
  }

  float warp_sum = warpReduceSum(local_sum_sq);

  if (lane_id == 0) {
    warp_sums[warp_id] = warp_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = lane_id < NUM_WARPS ? warp_sums[lane_id] : 0.0f;
    block_sum = warpReduceSum(block_sum);

    if (lane_id == 0) {
      s_inv_rms = rsqrtf(block_sum / static_cast<float>(HIDDEN_DIM) + eps);
    }
  }
  __syncthreads();

  const float inv_rms = s_inv_rms;

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int col = tid + i * BLOCK_SIZE;

    if (col < HIDDEN_DIM) {
      const float x = ScalarOps<T>::toFloat(x_reg[i]);
      const float w = ScalarOps<T>::toFloat(weight[col]);
      output[row_offset + col] = ScalarOps<T>::fromFloat(x * inv_rms * w);
    }
  }
}

//

// kernel launch 模版
template <typename T, int HIDDEN_DIM, int BLOCK_SIZE>
void launchRmsNormV5(const T *d_input, const T *d_weight, T *d_output,
                     size_t rows, float eps) {
  constexpr int VEC_WIDTH = VecTraits<T>::WIDTH;

  if constexpr (HIDDEN_DIM % VEC_WIDTH == 0) {
    rmsNormVectorKernelV5<T, HIDDEN_DIM, BLOCK_SIZE>
        <<<rows, BLOCK_SIZE>>>(d_input, d_weight, d_output, rows, eps);

  } else {
    rmsNormScalarKernelV5<T, HIDDEN_DIM, BLOCK_SIZE>
        <<<rows, BLOCK_SIZE>>>(d_input, d_weight, d_output, rows, eps);
  }
}

// 运行时根据hidden_dim值，选取相应特化版本。若有新的hidden_dim值出现，依据其是否满足向量化访问的对齐/整除要求，添加对应的特化版本。
template <typename T>
void dispatchRmsNormV5(const T *d_input, const T *d_weight, T *d_output,
                       size_t rows, size_t hidden_dim, float eps) {

  switch (hidden_dim) {
  case 1:
    launchRmsNormV5<T, 1, 32>(d_input, d_weight, d_output, rows, eps);
    break;

  case 8:
    launchRmsNormV5<T, 8, 32>(d_input, d_weight, d_output, rows, eps);
    break;

  case 16:
    launchRmsNormV5<T, 16, 32>(d_input, d_weight, d_output, rows, eps);
    break;

  case 31:
    launchRmsNormV5<T, 31, 32>(d_input, d_weight, d_output, rows, eps);
    break;

  case 64:
    launchRmsNormV5<T, 64, 32>(d_input, d_weight, d_output, rows, eps);
    break;

  case 128:
    launchRmsNormV5<T, 128, 32>(d_input, d_weight, d_output, rows, eps);
    break;

  case 256:
    launchRmsNormV5<T, 256, 64>(d_input, d_weight, d_output, rows, eps);
    break;

  case 512:
    launchRmsNormV5<T, 512, 128>(d_input, d_weight, d_output, rows, eps);
    break;

  case 769:
    launchRmsNormV5<T, 769, 256>(d_input, d_weight, d_output, rows, eps);
    break;

  case 1024:
    launchRmsNormV5<T, 1024, 256>(d_input, d_weight, d_output, rows, eps);
    break;

  case 1536:
    launchRmsNormV5<T, 1536, 256>(d_input, d_weight, d_output, rows, eps);
    break;

  case 2048:
    launchRmsNormV5<T, 2048, 256>(d_input, d_weight, d_output, rows, eps);
    break;

  case 4096:
    launchRmsNormV5<T, 4096, 256>(d_input, d_weight, d_output, rows, eps);
    break;

  default:
    break;
  }
}

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T> &h_input, const std::vector<T> &h_weight,
             std::vector<T> &h_output, size_t rows, size_t hidden_dim,
             float eps) {
  // TODO: Implement the rmsNorm function

  const size_t num_elements = rows * hidden_dim;
  const size_t input_bytes = num_elements * sizeof(T);
  const size_t weight_bytes = hidden_dim * sizeof(T);
  const size_t output_bytes = num_elements * sizeof(T);

  T *d_input = nullptr;
  T *d_weight = nullptr;
  T *d_output = nullptr;

  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_input), input_bytes));
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_weight), weight_bytes));
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes));

  RUNTIME_CHECK(
      cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes,
                           cudaMemcpyHostToDevice));

  dispatchRmsNormV5<T>(d_input, d_weight, d_output, rows, hidden_dim, eps);

  RUNTIME_CHECK(cudaGetLastError());

  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, output_bytes,
                           cudaMemcpyDeviceToHost));

  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
}

//---------------------------------------------------------------------------------
// 定义两个蝶形线程束规约函数，最终所有 lane 都拥有完整结果，用于online
// softmax阶段计算running max(m)和running sum(l)
__device__ __forceinline__ float warpButterflyReduceMax(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }

  return val;
}

__device__ __forceinline__ float warpButterflyReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }

  return val;
}

// 当前线程映射：
//
// 1 block = 4 warp
// 1 warp  = 1 query
// 1 lane  = 当前 KV tile 中的 1 个 key
//
// BLOCK_M = 4
// BLOCK_N = 32
//
// 因此一个 block 每次计算：
//
//      Q : [4, D]
//      K : [32, D]
//      V : [32, D]
//
//      S = QK^T : [4, 32]
//
// S 不写入 shared/global，
// 每一行分散在一个 warp 的 32 个 lane 寄存器中

template <int MAX_HEAD_DIM, int BLOCK_M = 4, int BLOCK_N = 32>
__global__ void flashAttentionFloatKernel(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, float *__restrict__ o, int target_seq_len,
    int src_seq_len, int query_heads, int kv_heads, int head_dim,
    bool is_causal, double softmax_scale) {
  // 线程映射
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  // 线程块映射
  // x -> Q tile
  // y -> query head
  // z -> batch
  const int q_start = blockIdx.x * BLOCK_M;
  const int q_idx = q_start + warp_id;
  const int query_head = blockIdx.y;
  const int batch = blockIdx.z;

  const bool q_valid = q_idx < target_seq_len;

  // GQA/MQA 映射
  // eg: Hq = 8, Hkv = 2
  // q head 0~3 -> kv head 0
  // q head 4~7 -> kv head 1
  const int group_size = query_heads / kv_heads;
  const int kv_head = query_head / group_size;

  // 共享内存定义
  // Q:[BLOCK_M][MAX_HEAD_DIM]
  // K:[MAX_HEAD_DIM][BLOCK_N + 1] K 在 shared 中转置
  // V:[BLOCK_N][MAX_HEAD_DIM]
  __shared__ float q_shared[BLOCK_M][MAX_HEAD_DIM];
  __shared__ float k_shared[MAX_HEAD_DIM][BLOCK_N + 1];
  __shared__ float v_shared[BLOCK_N][MAX_HEAD_DIM];

  // 从HBM中加载Q tile到Register中
  // 对于每个block来说，Q 只需要加载一次，后面不断复用它去扫描所有 K/V tile
  const int q_tile_elements = BLOCK_M * head_dim;

  for (int idx = tid; idx < q_tile_elements;
       idx +=
       blockDim.x) { // 为什么不是用block内所有线程来做？tid不是最多到32吗？
    const int local_q = idx / head_dim;
    const int d = idx % head_dim;
    const int global_q = q_start + local_q;

    float q_value = 0.0f;

    if (global_q < target_seq_len) {
      const size_t q_offset =
          (((size_t)batch * target_seq_len + global_q) * query_heads +
           query_head) *
              head_dim +
          d;

      q_value = q[q_offset]; // q[batch][global_q][query_head][d];
                             // ??????????batch和target_seq_len会变化吗？
                             // Head怎么并行的？
    }

    q_shared[local_q][d] = q_value;
  }

  __syncthreads();

  // Online softmax过程中需要维护的两个标量
  // 每个warp对应一个query
  // m_i:到目前为止这一行的最大score
  // l_i:到目前为止这一行的exp sum
  // 同一个warp的所有lane的 m_i / l_i始终保持一致
  float m_i = -CUDART_INF_F;
  float l_i = 0.0f;

  // Online softmax过程中需要计算的未归一化输出O
  // 因为每个线程计算出的未归一化输出O无需与其他线程共享且所占字节数较少，所以由线程私用寄存器来进行存储
  // eg: MAX_HEAD_DIM = 64
  // lane0:O[0] O[32]
  // lane1:O[1] O[33]
  // ...
  constexpr int OUTPUTS_PER_LANE = (MAX_HEAD_DIM + 32 - 1) / 32;
  float o_reg[OUTPUTS_PER_LANE];

#pragma unroll
  for (int item = 0; item < OUTPUTS_PER_LANE; ++item) {
    o_reg[item] = 0.0f;
  }

  // 扫描K/V tiles
  for (int k_start = 0; k_start < src_seq_len; k_start += BLOCK_N) {
    // 如果is_causal，则不必继续扫描K/V tiles
    if (is_causal) {
      int max_q = q_start + BLOCK_M - 1;

      if (max_q >= target_seq_len) {
        max_q = target_seq_len - 1;
      }

      if (k_start > max_q) {
        break;
      }
    }

    // 从HBM中加载K tile[32, D]和V tile[32, D]到shared memory中
    const int kv_tile_elements = BLOCK_N * head_dim;

    for (int idx = tid; idx < kv_tile_elements; idx += blockDim.x) {
      const int local_k = idx / head_dim;
      const int d = idx % head_dim;
      const int global_k = k_start + local_k;

      float k_value = 0.0f;
      float v_value = 0.0f;

      if (global_k < src_seq_len) {
        const size_t kv_offset =
            (((size_t)batch * src_seq_len + global_k) * kv_heads + kv_head) *
                head_dim +
            d;

        k_value = k[kv_offset];
        v_value = v[kv_offset];
      }
      k_shared[d][local_k] = k_value;
      v_shared[local_k][d] = v_value;
    }
    __syncthreads();

    if (q_valid) {
      // 当前lane对应当前tile中的一个key
      const int key_idx = k_start + lane_id;
      bool key_valid = key_idx < src_seq_len; // 何意

      // 若j>i则masked
      if (is_causal && key_idx > q_idx) {
        key_valid = false;
      }

      // Q@K^T
      // 一个lane负责score_j = Q_i · K_j
      float score_acc = 0.0f;

      if (key_valid) {
        for (int d = 0; d < head_dim; ++d) {
          const float q_value = q_shared[warp_id][d];
          const float k_value = k_shared[d][lane_id];

          score_acc = fmaf(q_value, k_value, score_acc);
        }
      }

      // 为通过#6 #14测试用例，进行精度增强
      // QK accumulation : float
      // scale: double
      // scaled score: double
      const double score =
          key_valid ? static_cast<double>(score_acc) * softmax_scale : 0.0;

      // warp max 仍然保持 float
      const float score_for_max =
          key_valid ? static_cast<float>(score) : -CUDART_INF_F;

      // Tile Softmax
      // 求当前tile的行最大值m_tile
      const float m_tile = warpButterflyReduceMax(score_for_max);

      // 如果当前整个tile都被mask, 则直接跳过；
      if (m_tile != -CUDART_INF_F) {
        const float m_new = fmaxf(m_i, m_tile); // 更新全局running max
        // const float p =
        //     key_valid ? expf(score - m_tile) : 0.0f;    // 计算当前tile的exp
        //     弃用 使用FA2的更新方式
        // const float l_tile = warpButterflyReduceSum(p); // 计算当前tile的sum
        // 弃用 使用FA2的更新方式
        const float alpha =
            m_i == -CUDART_INF_F
                ? 0.0f
                : expf(
                      m_i -
                      m_new); // alpha用于对之前的tile的l_tile_old和O_old进行重缩放修正
        // const float beta = expf(
        //     m_tile - m_new); // 弃用 使用FA2的更新方式

        // const float p = key_valid ? expf(score - m_new) : 0.0f; 精度不够
        const float p =
            key_valid
                ? static_cast<float>(exp(score - static_cast<double>(m_new)))
                : 0.0f;
        const float l_tile = warpButterflyReduceSum(p); // 计算当前tile的sum

        // 使用alpha对O_old进行重缩放修正
#pragma unroll
        for (int item = 0; item < OUTPUTS_PER_LANE; ++item) {
          const int d = lane_id + item * 32;

          if (d < head_dim) {
            o_reg[item] *= alpha;
          }
        }
        // const float p_scaled = p * beta; // 弃用 使用FA2的更新方式

// P @ V
// 当前每个lane的寄存器存储了当前P tile行(BN)的一部分，合起来是P tile的一整行
// lane0 存储 p0, lane1 存储 p1，...，lane31 存储 p31
// 如果每个lane需要计算出O = P@V
// 中的一个输出元素的部分和O_partialSum，则它需要所有其它线程束内线程存储的p信息
// 这可以通过洗牌函数来实现
#pragma unroll
        for (int j = 0; j < BLOCK_N; ++j) {
          const float weight = __shfl_sync(0xffffffff, p, j);

// 一个warp负责一行输出元素O，则每个lane需要负责计算D / BN次
#pragma unroll
          for (int item = 0; item < OUTPUTS_PER_LANE; ++item) {
            const int d = lane_id + item * 32;

            if (d < head_dim) {
              const float v_value = v_shared[j][d];
              o_reg[item] = fmaf(weight, v_value, o_reg[item]);
            }
          }
        }

        // 更新 online softmax 的running sum 和 max
        l_i = alpha * l_i + l_tile;
        m_i = m_new;
      }
    }
    __syncthreads();
  }

  // Online Softmax 收尾,利用最终的sum l，归一化输出O
  if (q_valid) {
    const float inv_l = l_i > 0.0f ? 1.0f / l_i : 0.0f;
#pragma unroll
    for (int item = 0; item < OUTPUTS_PER_LANE; ++item) {
      const int d = lane_id + item * 32;

      if (d < head_dim) {
        const size_t o_offset =
            (((size_t)batch * target_seq_len + q_idx) * query_heads +
             query_head) *
                head_dim +
            d;

        o[o_offset] = o_reg[item] * inv_l;
      }
    }
  }
}

// 大体上同float版,不同的地方在于精度
// Q/K/V:FP16
// QK multiplication:FP16 CUDA Core
// QK accumulation:FP32
// softmax:FP32
// P × V multiplication:FP16 CUDA Core
// O accumulation:FP32
// output:FP16
template <int MAX_HEAD_DIM, int BLOCK_M = 4, int BLOCK_N = 32>
__global__ void
flashAttentionHalfKernel(const half *__restrict__ q, const half *__restrict__ k,
                         const half *__restrict__ v, half *__restrict__ o,
                         int target_seq_len, int src_seq_len, int query_heads,
                         int kv_heads, int head_dim, bool is_causal,
                         float softmax_scale) {
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  const int q_start = blockIdx.x * BLOCK_M;
  const int q_idx = q_start + warp_id;
  const int query_head = blockIdx.y;
  const int batch = blockIdx.z;

  const bool q_valid = q_idx < target_seq_len;

  // GQA / MQA
  const int group_size = query_heads / kv_heads;
  const int kv_head = query_head / group_size;

  // Shared Memory
  __shared__ half q_shared[BLOCK_M][MAX_HEAD_DIM];
  __shared__ half k_shared[MAX_HEAD_DIM][BLOCK_N + 1];
  __shared__ half v_shared[BLOCK_N][MAX_HEAD_DIM];

  // Load Q
  const int q_tile_elements = BLOCK_M * head_dim;

  for (int idx = tid; idx < q_tile_elements; idx += blockDim.x) {
    const int local_q = idx / head_dim;
    const int d = idx % head_dim;
    const int global_q = q_start + local_q;

    half q_value = __float2half_rn(0.0f);

    if (global_q < target_seq_len) {
      const size_t q_offset =
          (((size_t)batch * target_seq_len + global_q) * query_heads +
           query_head) *
              head_dim +
          d;

      q_value = q[q_offset];
    }

    q_shared[local_q][d] = q_value;
  }
  __syncthreads();

  // Online Softmax
  float m_i = -CUDART_INF_F;
  float l_i = 0.0f;

  constexpr int OUTPUTS_PER_LANE = (MAX_HEAD_DIM + 31) / 32;
  float o_reg[OUTPUTS_PER_LANE];

#pragma unroll
  for (int item = 0; item < OUTPUTS_PER_LANE; ++item) {
    o_reg[item] = 0.0f;
  }

  // 扫描 K/V tiles
  for (int k_start = 0; k_start < src_seq_len; k_start += BLOCK_N) {
    // causal判定
    if (is_causal) {
      int max_q = q_start + BLOCK_M - 1;
      if (max_q >= target_seq_len) {
        max_q = target_seq_len - 1;
      }

      if (k_start > max_q) {
        break;
      }
    }

    //  加载K/V
    const int kv_tile_elements = BLOCK_N * head_dim;

    for (int idx = tid; idx < kv_tile_elements; idx += blockDim.x) {
      const int local_k = idx / head_dim;
      const int d = idx % head_dim;
      const int global_k = k_start + local_k;

      half k_value = __float2half_rn(0.0f);
      half v_value = __float2half_rn(0.0f);

      if (global_k < src_seq_len) {
        const size_t kv_offset =
            (((size_t)batch * src_seq_len + global_k) * kv_heads + kv_head) *
                head_dim +
            d;

        k_value = k[kv_offset];
        v_value = v[kv_offset];
      }

      // 转置K
      k_shared[d][local_k] = k_value;
      v_shared[local_k][d] = v_value;
    }

    __syncthreads();

    // 计算仍像float版本 One warp -> one query

    if (q_valid) {
      const int key_idx = k_start + lane_id;
      bool key_valid = key_idx < src_seq_len;

      if (is_causal && key_idx > q_idx) {
        key_valid = false;
      }

      // Q @ K^T
      float score_acc = 0.0f;

      if (key_valid) {

#pragma unroll
        for (int d = 0; d < head_dim; ++d) {
          const float q_value = __half2float(q_shared[warp_id][d]);
          const float k_value = __half2float(k_shared[d][lane_id]);

          score_acc = fmaf(q_value, k_value, score_acc);
        }
      }

      // Scale
      const float score = key_valid ? score_acc * softmax_scale : 0.0f;
      const float score_for_max = key_valid ? score : -CUDART_INF_F;

      // Tile Softmax
      const float m_tile = warpButterflyReduceMax(score_for_max);

      if (m_tile != -CUDART_INF_F) {
        const float m_new = fmaxf(m_i, m_tile);
        const float alpha = m_i == -CUDART_INF_F ? 0.0f : expf(m_i - m_new);
        const float p = key_valid ? expf(score - m_new) : 0.0f;
        const float l_tile = warpButterflyReduceSum(p);

#pragma unroll
        for (int item = 0; item < OUTPUTS_PER_LANE; ++item) {
          const int d = lane_id + item * 32;

          if (d < head_dim) {
            o_reg[item] *= alpha;
          }
        }

        // P @ V
#pragma unroll
        for (int j = 0; j < BLOCK_N; ++j) {

          const float weight = __shfl_sync(0xffffffff, p, j);

#pragma unroll
          for (int item = 0; item < OUTPUTS_PER_LANE; ++item) {
            const int d = lane_id + item * 32;

            if (d < head_dim) {
              const float v_value = __half2float(v_shared[j][d]);

              o_reg[item] = fmaf(weight, v_value, o_reg[item]);
            }
          }
        }

        l_i = alpha * l_i + l_tile;
        m_i = m_new;
      }
    }

    __syncthreads();
  }

  // O

  if (q_valid) {
    const float inv_l = l_i > 0.0f ? 1.0f / l_i : 0.0f;

#pragma unroll
    for (int item = 0; item < OUTPUTS_PER_LANE; ++item) {
      const int d = lane_id + item * 32;

      if (d < head_dim) {
        const size_t o_offset =
            (((size_t)batch * target_seq_len + q_idx) * query_heads +
             query_head) *
                head_dim +
            d;
        const float result = o_reg[item] * inv_l;

        o[o_offset] = __float2half_rn(result);
      }
    }
  }
}

void launchFlashAttentionFloat(const float *d_q, const float *d_k,
                               const float *d_v, float *d_o, int batch_size,
                               int target_seq_len, int src_seq_len,
                               int query_heads, int kv_heads, int head_dim,
                               bool is_causal) {
  constexpr int BLOCK_M = 4;

  dim3 block(BLOCK_M * 32);
  dim3 grid((target_seq_len + BLOCK_M - 1) / BLOCK_M, query_heads, batch_size);

  const double softmax_scale = 1.0f / sqrtf(static_cast<double>(head_dim));

  if (head_dim <= 32) {
    flashAttentionFloatKernel<32><<<grid, block>>>(
        d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
        head_dim, is_causal, softmax_scale);

  } else if (head_dim <= 64) {
    flashAttentionFloatKernel<64><<<grid, block>>>(
        d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
        head_dim, is_causal, softmax_scale);

  } else {
    flashAttentionFloatKernel<128><<<grid, block>>>(
        d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
        head_dim, is_causal, softmax_scale);
  }

  RUNTIME_CHECK(cudaGetLastError());
}

void launchFlashAttentionHalf(const half *d_q, const half *d_k, const half *d_v,
                              half *d_o,

                              int batch_size, int target_seq_len,
                              int src_seq_len, int query_heads, int kv_heads,
                              int head_dim, bool is_causal) {

  constexpr int BLOCK_M = 4;

  dim3 block(BLOCK_M * 32);
  dim3 grid((target_seq_len + BLOCK_M - 1) / BLOCK_M, query_heads, batch_size);

  const float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  if (head_dim <= 32) {

    flashAttentionHalfKernel<32><<<grid, block>>>(
        d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
        head_dim, is_causal, softmax_scale);

  } else if (head_dim <= 64) {

    flashAttentionHalfKernel<64><<<grid, block>>>(
        d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
        head_dim, is_causal, softmax_scale);

  } else {
    flashAttentionHalfKernel<128><<<grid, block>>>(
        d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
        head_dim, is_causal, softmax_scale);
  }

  RUNTIME_CHECK(cudaGetLastError());
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len,
 * query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size,
 * tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query
 * attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  // TODO: Implement the flash attention function

  const size_t q_elements =
      static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t kv_elements =
      static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;

  T *d_q = nullptr;
  T *d_k = nullptr;
  T *d_v = nullptr;
  T *d_o = nullptr;

  RUNTIME_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_q), q_elements * sizeof(T)));
  RUNTIME_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_k), kv_elements * sizeof(T)));
  RUNTIME_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_v), kv_elements * sizeof(T)));
  RUNTIME_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_o), q_elements * sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_elements * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_elements * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_elements * sizeof(T),
                           cudaMemcpyHostToDevice));

  if constexpr (std::is_same_v<T, float>) {

    launchFlashAttentionFloat(d_q, d_k, d_v, d_o, batch_size, target_seq_len,
                              src_seq_len, query_heads, kv_heads, head_dim,
                              is_causal);

  } else if constexpr (std::is_same_v<T, half>) {
    launchFlashAttentionHalf(d_q, d_k, d_v, d_o, batch_size, target_seq_len,
                             src_seq_len, query_heads, kv_heads, head_dim,
                             is_causal);
  }

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, q_elements * sizeof(T),
                           cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float> &,
                             const std::vector<float> &, std::vector<float> &,
                             size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half> &,
                            const std::vector<half> &, std::vector<half> &,
                            size_t, size_t, float);
template void flashAttention<float>(const std::vector<float> &,
                                    const std::vector<float> &,
                                    const std::vector<float> &,
                                    std::vector<float> &, int, int, int, int,
                                    int, int, bool);
template void flashAttention<half>(const std::vector<half> &,
                                   const std::vector<half> &,
                                   const std::vector<half> &,
                                   std::vector<half> &, int, int, int, int, int,
                                   int, bool);
