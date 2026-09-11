#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h> 
#include "osCfar.h"
#include <sys/time.h>
#include <omp.h>
#include <sched.h>
#include <pthread.h>
#include <immintrin.h>
#include <limits.h>
#include <stdint.h>   
#include <float.h> 
#include <mkl_vsl.h>
#include <mkl_vml.h>
#include <time.h>
#include <stdbool.h>
#include <mkl.h>
#include <errno.h>
#include <stddef.h>
#include "osCfarOpt.h"
#include <assert.h>

#define FORCE_INLINE __attribute__((always_inline))

 double timeval(struct timeval* start, struct timeval* end) {
    long seconds = end->tv_sec - start->tv_sec;
    long microseconds = end->tv_usec - start->tv_usec;
    return seconds + microseconds / 1000000.0;
}

inline void arrayRoll(const float *in, float *out, int len, int shift) {
    shift = (shift % len + len) % len;
    for (int i = 0; i < len; i++) {
        int idx = (i - shift + len) % len;
        out[i] = in[idx];
    }
}

inline void localMaxDetect(const float *data, int len, bool *out) {
    float *shiftR1 = (float *)malloc(len * sizeof(float));
    float *shiftL1 = (float *)malloc(len * sizeof(float));

    /*Performs a right circular shift by 1 position on the array ComparatorIn: The rightmost element of the array is moved to the leftmost position, and all other elements are shifted right by 1 position*/
    arrayRoll(data, shiftR1, len, 1);
    /*Performs a left circular shift by 1 position on the array ComparatorIn: The leftmost element of the array is moved to the rightmost position, and all other elements are shifted left by 1 position*/
    arrayRoll(data, shiftL1, len, -1);
    /*Non-Maximum Suppression*/
    for (int i = 0; i < len; i++) {
        if ((data[i] >= shiftR1[i]) && (data[i] >= shiftL1[i])) {
            out[i] = true;
        }
        else {
            out[i] = false;
        }
    }
    free(shiftR1);
    free(shiftL1);
}

int compare_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

float avx2_sort_128f(const float *src)
{
    float buf[128];
    for (int i = 0; i < 16; i++) {  // 16 × 8 = 128
        _mm256_storeu_ps(buf + i*8, _mm256_loadu_ps(src + i*8));
    }

    int left = 0, right = 127;
    const int target = 63;

    while (left < right)
    {
        float pivot = buf[(left + right) / 2];
        int i = left, j = right;

        while (i <= j)
        {
            while (buf[i] < pivot) i++;
            while (buf[j] > pivot) j--;

            if (i <= j)
            {
                float tmp = buf[i];
                buf[i] = buf[j];
                buf[j] = tmp;
                i++; j--;
            }
        }

        if (target <= j)
            right = j;
        else if (target >= i)
            left = i;
        else
            break;
    }
    return buf[target];
}

void osCfarNoiseFinder_Opt(const float *data, int col, CFARConfig* config, bool *isValid, float *noiseOut) {
    int winLen = config->CFAR_AVG_LEFT + config->CFAR_AVG_RIGHT;
    float win[config->CFAR_AVG_LEFT + config->CFAR_AVG_RIGHT];
    for (int i=0; i<col; i++) {
       // if (true == isValid[i]) {
            for (int leftIdx=0; leftIdx<config->CFAR_AVG_LEFT; leftIdx++) {
                int curLeftIdx = i - config->CFAR_AVG_LEFT - config->CFAR_GUARD_INT + leftIdx;
                if (curLeftIdx < 0) {
                    curLeftIdx += col;
                }
                
                if (curLeftIdx >=0 ) win[leftIdx] = data[curLeftIdx];
            }

            for (int rightIdx=0; rightIdx<config->CFAR_AVG_RIGHT; rightIdx++) {
                int curRightIdx = i + config->CFAR_GUARD_INT + rightIdx + 1;
                if (curRightIdx >= col) {
                    curRightIdx -= col;
                }
                
                if ((config->CFAR_AVG_LEFT + rightIdx < winLen) && curRightIdx >= 0) win[config->CFAR_AVG_LEFT + rightIdx] = data[curRightIdx];
            } 
            //qsort(win, winLen, sizeof(float), compare_float);
            //noiseOut[i] = win[config->CFAR_OS_KVALUE];
            noiseOut[i] =  avx2_sort_128f(win);
       // }
    }
}

#include <immintrin.h>
#include <string.h>

// 正确的8元素双调排序（无宏，直接实现）
static inline __m256 bitonic_sort8_correct(__m256 v) {
    __m256 t;
    __m256 minv, maxv;
    
    // 第1层
    t = _mm256_permute_ps(v, 0xB1);
    minv = _mm256_min_ps(v, t);
    maxv = _mm256_max_ps(v, t);
    v = _mm256_blend_ps(minv, maxv, 0xAA);
    
    // 第2层
    t = _mm256_permute_ps(v, 0x4E);
    minv = _mm256_min_ps(v, t);
    maxv = _mm256_max_ps(v, t);
    v = _mm256_blend_ps(minv, maxv, 0xCC);
    
    // 第3层
    t = _mm256_permute_ps(v, 0xB1);
    minv = _mm256_min_ps(v, t);
    maxv = _mm256_max_ps(v, t);
    v = _mm256_blend_ps(minv, maxv, 0xAA);
    
    // 第4层
    t = _mm256_permute_ps(v, 0x1B);
    minv = _mm256_min_ps(v, t);
    maxv = _mm256_max_ps(v, t);
    v = _mm256_blend_ps(minv, maxv, 0xF0);
    
    // 第5层
    t = _mm256_permute_ps(v, 0xB1);
    minv = _mm256_min_ps(v, t);
    maxv = _mm256_max_ps(v, t);
    v = _mm256_blend_ps(minv, maxv, 0xAA);
    
    // 第6层
    t = _mm256_permute_ps(v, 0x4E);
    minv = _mm256_min_ps(v, t);
    maxv = _mm256_max_ps(v, t);
    v = _mm256_blend_ps(minv, maxv, 0xCC);
    
    // 第7层
    t = _mm256_permute_ps(v, 0xB1);
    minv = _mm256_min_ps(v, t);
    maxv = _mm256_max_ps(v, t);
    v = _mm256_blend_ps(minv, maxv, 0xAA);
    
    return v;
}

void os_cfar_avx512_vertical_float(const float* padded_row, const float* cut_row, int cols, const CFARConfig* config, int* results) {
    const int h_win = config->CFAR_AVG_LEFT;
    const int h_g = config->CFAR_GUARD_INT;
    const float offset = config->CFAR_THR_KSCALE;
    const int k_rank = config->CFAR_OS_KVALUE;

    __m512  v_offset = _mm512_set1_ps(offset);
    __m512  v_zero   = _mm512_setzero_ps();
    __m512i v_k      = _mm512_set1_epi32(k_rank - 1);
    __m512i v_one    = _mm512_set1_epi32(1);

    int i = 0;
    for (; i <= cols - 16; i += 16) {
        __m512 v_cuts = _mm512_loadu_ps(&cut_row[i]);
        
        // 关键修复：v_thr = max(cur - offset, 0)
        __m512 v_thr = _mm512_max_ps(_mm512_sub_ps(v_cuts, v_offset), v_zero);

        __m512i v_counts = _mm512_setzero_epi32();

        /*for (int off = -(h_win + h_g); off <= (h_win + h_g); off++) {
            if (off >= -h_g && off <= h_g)
                continue;

            __m512 v_ref = _mm512_loadu_ps(&padded_row[i + h_win + h_g + off]);
            __mmask16 mask = _mm512_cmp_ps_mask(v_thr, v_ref, _CMP_GT_OQ);
            v_counts = _mm512_add_epi32(v_counts, _mm512_maskz_set1_epi32(mask, 1));
        }*/

        for (int off = -(h_win + h_g); off < -h_g; off++) {
            __m512 v_ref = _mm512_loadu_ps(&padded_row[i + h_win + h_g + off]);
            __mmask16 mask = _mm512_cmp_ps_mask(v_thr, v_ref, _CMP_GT_OQ);
            v_counts = _mm512_add_epi32(v_counts, _mm512_maskz_set1_epi32(mask, 1));
        }

        for (int off = h_g+1; off <= (h_win + h_g); off++) {
            __m512 v_ref = _mm512_loadu_ps(&padded_row[i + h_win + h_g + off]);
            __mmask16 mask = _mm512_cmp_ps_mask(v_thr, v_ref, _CMP_GT_OQ);
            v_counts = _mm512_add_epi32(v_counts, _mm512_maskz_set1_epi32(mask, 1));
        }

        __mmask16 res_mask = _mm512_cmp_epi32_mask(v_counts, v_k, _MM_CMPINT_GT);
        __m512i v_res = _mm512_maskz_set1_epi32(res_mask, 1);
        _mm512_storeu_epi32(&results[i], v_res);
    }

    // 标量收尾（也需要修复）
    for (; i < cols; i++) {
        float cur = cut_row[i];
        float v_thr = (cur > offset) ? (cur - offset) : 0;  // 修复
        int count = 0;

        for (int off = -(h_win + h_g); off <= (h_win + h_g); off++) {
            if (off >= -h_g && off <= h_g)
                continue;
            if (padded_row[i + h_win + h_g + off] < v_thr)
                count++;
        }
        results[i] = (count >= k_rank) ? 1 : 0;
    }
}

void stable_sort_float(float *arr, int n) {
    for (int i = 1; i < n; i++) {
        float key = arr[i];
        int j = i - 1;
        // 从小到大排序（和 np.sort 一致）
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void osCfarNoiseFinder(const float *data, int col, CFARConfig* config, float *noiseOut) {
    const int guard_total = 2 * config->CFAR_GUARD_INT + 1;    // =9
    const int left = config->CFAR_AVG_LEFT;                    // 20
    const int right = config->CFAR_AVG_RIGHT;                  // 20
    const int win_total = left + right;                // 40
    const int offset = left + guard_total;
    int windows_cnt = col - 2*(config->CFAR_GUARD_INT + config->CFAR_AVG_LEFT);
       
    int* idx_left = (int*)mkl_malloc(windows_cnt * left * sizeof(int), 64);
    int* idx_right = (int*)mkl_malloc(windows_cnt * right * sizeof(int), 64);
    float* stat_win = (float*)mkl_malloc(windows_cnt * win_total * sizeof(float), 64);
        
    for (int j = 0; j < left; j++) {
        for (int i = 0; i < windows_cnt; i++) {
            idx_left[i + j * windows_cnt] = j+i;
        }
    }

    for (int j = 0; j < right; j++) {
        for (int i = 0; i < windows_cnt; i++) {
            idx_right[i + j * windows_cnt] = j + offset+i;
        }
    }

    cblas_sgthr(windows_cnt * left, data, stat_win, idx_left);
    cblas_sgthr(windows_cnt * right, data, stat_win + windows_cnt * left, idx_right);
     
    for (int i = 0; i < windows_cnt; i++) {
        stable_sort_float(stat_win + i * win_total, win_total);
        noiseOut[i] = stat_win[config->CFAR_OS_KVALUE];
      //if (true == isValid[i])noiseOut[i] = avx2_sort_128f(stat_win + i * win_total);
    }

    mkl_free(idx_left);
    mkl_free(idx_right);
    mkl_free(stat_win);
}

void extendMatrix(const float* dataMatrix, CFARConfig* config, int rows, int cols, float* dataMatrixExt)
{
    const int ext = config->CFAR_GUARD_INT + config->CFAR_AVG_LEFT;  
    const int new_cols = ext + cols + ext;  //
    // 左扩展：拷贝所有行的最后68列
    mkl_somatcopy('R', 'N', rows, ext,
                  1.0f,
                  dataMatrix + (cols - ext), cols,
                  dataMatrixExt, new_cols);

    // 中间：拷贝原始矩阵全部数据
    mkl_somatcopy('R', 'N', rows, cols,
                  1.0f,
                  dataMatrix, cols,
                  dataMatrixExt + ext, new_cols);

    // 右扩展：拷贝所有行的前68列
    mkl_somatcopy('R', 'N', rows, ext,
                  1.0f,
                  dataMatrix, cols,
                  dataMatrixExt + ext + cols, new_cols);
}

/* This method implements the 1D CFAR Detection function
 input: dbData --- a dbData vector for scanning, notice that is vector is not enriched for the cyclic case
 input: config --- cfar configuration factors
 input: row -- number of row, for example 512 or 1024
 input: col --- number of columns for example 512 or 1024
 output bitMap ---- target location
 output: noiseMap --- noise vector corresponding to each "real" data element*/
inline int find_max_index(const float *src, int n, float *averageValue)
{
    if (n <= 0) return -1;

    int max_idx = 0;
    float max_val = src[0];
    float sum = src[0];

    for (int i = 1; i < n; i++) {
        sum +=src[i];
        if (src[i] > max_val) {
            max_val = src[i];
            max_idx = i;
        }
    }
    *averageValue = sum/n;
    return max_idx;
}

bool *localMax_buf = NULL;
#pragma omp threadprivate(localMax_buf)

// 内存对齐宏（保留，有用）
#define ALIGNMENT 64
#define ALIGN_MALLOC(ptr, size) posix_memalign((void**)&ptr, ALIGNMENT, size)
#define BLOCK_SIZE  8
 void matriTransponse(float *in, float *out, int row, int col) {
    for (int ii = 0; ii < row; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < col; jj += BLOCK_SIZE) {
            for (int i = ii; i < ii + BLOCK_SIZE && i < row; i++) {
                for (int j = jj; j < jj + BLOCK_SIZE && j < col; j++) {
                    out[i*col+j] = in[j*row+i];
                }
            }
        }
    }
}

#define BOOL_BLOCK_SIZE 64
//161*241转置为241*161
void boolMatriTransponse(bool *in, bool *out) {
    for (int jj = 0; jj < Y_RANGE; jj += BOOL_BLOCK_SIZE) {
        for (int ii = 0; ii < X_RANGE; ii += BOOL_BLOCK_SIZE) {
            for (int j = jj; j < jj + BOOL_BLOCK_SIZE && j < Y_RANGE; j++) {
                for (int i = ii; i < ii + BOOL_BLOCK_SIZE && i < X_RANGE; i++) {
                    out[i*Y_RANGE+j] = in[j*X_RANGE+i];
                }
            }
        }
    }
}

/* This method implements the 2D CFAR Detection function: Detect targets in 1024 rows (512 columns each), 
  then transpose and detect in 512 rows (1024 columns each). Noise is the max of the row and column. 
  A target is valid only if it is detected in both the row and column detection results.
 input: dbData --- a dbData vector for scanning, notice that is vector is not enriched for the cyclic case
 input: config --- cfar configuration factors
 output bitMap ---- target location
 output: noiseMap --- noise vector corresponding to each "real" data element*/

int write_float_data(float (*all_rv_matrices)[Y_RANGE], const char* output_path, int frame_cnt, int Row, int Col) {
    FILE *fp = fopen(output_path, "w+");
    if (NULL == fp) {
        fprintf(stderr, "fopen %s failed: %d\n", output_path, strerror(errno));
        return -1;
    }

    for (int row=0; row<Row; row++) {
        for (int col=0; col<Col; col++) {
            int ret = fprintf(fp, "%.8f%c", all_rv_matrices[row][col], (col == 160)? '\n':' ');
            if (ret<0) {
                fprintf(stderr, "write frame %d row %d failed: %d\n", frame_cnt, row, strerror(errno));
                fclose(fp);
                return -1;
            }
        }
    }
    fflush(fp);
    fclose(fp);
    printf("All 25 frames processed and appended to %s\n", output_path);
    return 0;
}

static inline void transpose_8x8_avx2(const float *src, int src_stride, float *dst, int dst_stride) {
    // 加载8行
    __m256 r0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 r1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 r3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 r5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 r7 = _mm256_loadu_ps(src + 7 * src_stride);
    
    // 第1层：32位交换
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);
    
    // 第2层：64位交换
    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    
    // 第3层：128位交换并存储
    _mm256_storeu_ps(dst + 0 * dst_stride, _mm256_permute2f128_ps(s0, s4, 0x20));
    _mm256_storeu_ps(dst + 1 * dst_stride, _mm256_permute2f128_ps(s1, s5, 0x20));
    _mm256_storeu_ps(dst + 2 * dst_stride, _mm256_permute2f128_ps(s2, s6, 0x20));
    _mm256_storeu_ps(dst + 3 * dst_stride, _mm256_permute2f128_ps(s3, s7, 0x20));
    _mm256_storeu_ps(dst + 4 * dst_stride, _mm256_permute2f128_ps(s0, s4, 0x31));
    _mm256_storeu_ps(dst + 5 * dst_stride, _mm256_permute2f128_ps(s1, s5, 0x31));
    _mm256_storeu_ps(dst + 6 * dst_stride, _mm256_permute2f128_ps(s2, s6, 0x31));
    _mm256_storeu_ps(dst + 7 * dst_stride, _mm256_permute2f128_ps(s3, s7, 0x31));
}

void mat_transpose_avx2(
    const float *restrict src,
    float *restrict dst,
    int rows,    // 241
    int cols     // 161
) {
    const int dst_stride = rows;  // 转置后的列数 = 原行数
    for (int i = 0; i <= rows - 8; i += 8) {
        for (int j = 0; j <= cols - 8; j += 8) {
            transpose_8x8_avx2(
                src + i * cols + j, cols,      // 源：src[i][j]
                dst + j * dst_stride + i, dst_stride  // 目标：dst[j][i]
            );
        }
    }
    
    // 边界处理：右侧剩余列（j从cols-8到cols-1）
    int i_end = (rows / 8) * 8;
    int j_end = (cols / 8) * 8;
    
    // 处理右侧不完整的列
    for (int i = 0; i < rows; i++) {
        for (int j = j_end; j < cols; j++) {
            dst[j * dst_stride + i] = src[i * cols + j];
        }
    }
    
    // 处理底部不完整的行
    for (int i = i_end; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * dst_stride + i] = src[i * cols + j];
        }
    }
}

int write_float_bin_data(float (*all_rv_matrices)[Y_RANGE], const char* output_path, int frame_cnt, int row, int col) {
    FILE *fp = fopen(output_path, "wr+");
    if (NULL == fp) {
        fprintf(stderr, "fopen %s failed: %d\n", output_path, strerror(errno));
        return -1;
    }

    size_t total_bytes = sizeof(float)*row*col;
    size_t written_bytes = fwrite(all_rv_matrices, 1, total_bytes, fp);
    if (total_bytes != written_bytes) {
        fprintf(stderr, "write frame %d failed: %ld, total_bytes:%zu, written_bytes:%ld\n", frame_cnt, total_bytes, written_bytes);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

void assertMatTransfer(const float (* f_result)[X_RANGE][Y_RANGE], float (* dbDataTransponse)[Y_RANGE][X_RANGE], int z_idx) {
    /***************assert*********/
        for (int x_idx=0; x_idx < X_RANGE; x_idx++) {
            for (int y_idx=0; y_idx < Y_RANGE; y_idx++) {
               // printf("f_result[%d][%d][%d]=%.12f\n", z_idx, x_idx, y_idx, f_result[z_idx][x_idx][y_idx]);
               // printf("dbDataTransponse[%d][%d]=%.12f\n", z_idx, y_idx*X_RANGE + x_idx, dbDataTransponse[z_idx][y_idx][x_idx]);
                assert(f_result[z_idx][x_idx][y_idx]== dbDataTransponse[z_idx][y_idx][x_idx]);
            }
    }
}

void cfarDetect1d(const float* src, int col, CFARConfig* config, bool *bitMap) {
    int leftExt  = config->CFAR_AVG_LEFT + config->CFAR_GUARD_INT;
    int rightExt = config->CFAR_AVG_RIGHT + config->CFAR_GUARD_INT;
    int colExt = leftExt + col + rightExt;
    float* dataMatrixExt = (float*)mkl_malloc(colExt * sizeof(float), 64);
    extendMatrix(src, config, 1, col,  dataMatrixExt);
    assert(src[0] == dataMatrixExt[config->CFAR_AVG_LEFT + config->CFAR_GUARD_INT]);
    assert(src[0] == dataMatrixExt[config->CFAR_AVG_LEFT + config->CFAR_GUARD_INT + col]);
    assert(src[col -1] == dataMatrixExt[config->CFAR_AVG_LEFT + config->CFAR_GUARD_INT - 1]);
    assert(src[col -1] == dataMatrixExt[config->CFAR_AVG_LEFT + config->CFAR_GUARD_INT + col - 1]);

    bool localMax[col];
    bool isValid[col];
    float average = 0.0;

    int max_index = find_max_index(src, col, &average);
    //float noiseThreshMax = src[max_index] + config->CFAR_THR_SCALE;

    localMaxDetect(src, col, localMax);
    for (int i=0; i<col; i++) {
        isValid[i] = false;
        if ((true == localMax[i]) && ((src[i] >= average + 3))) {//(src[i] >= noiseThreshMax) && 
            isValid[i] = true;
        }
    }

    float noiseMapRow[col];
    int result[col];

    osCfarNoiseFinder(dataMatrixExt, colExt, config, noiseMapRow);
    for (int i=0; i<col; i++) {
        if ((true == isValid[i])&&(src[i] >= (noiseMapRow[i] + config->CFAR_THR_KSCALE))) {
            bitMap[i] = 1;
        }
        else {
            bitMap[i] = 0;
        }
    }

    mkl_free(dataMatrixExt);
}

void mat_zxy_transpose_xyz(const float (*inputData)[X_RANGE][Y_RANGE], float (*outData)[Y_RANGE][Z_RANGE]) {
    for (int z_idx=0; z_idx < Z_RANGE; z_idx++) {
        for (int x_idx=0; x_idx < X_RANGE; x_idx++) {
            for (int y_idx=0; y_idx < Y_RANGE; y_idx++) {
                outData[x_idx][y_idx][z_idx] = inputData[z_idx][x_idx][y_idx];
            }
        }
    }
}

void assert_zxy_transpose_xyz(const float (*inputData)[X_RANGE][Y_RANGE], float (*outData)[Y_RANGE][Z_RANGE]) {
    for (int z_idx=0; z_idx < Z_RANGE; z_idx++) {
        for (int x_idx=0; x_idx < X_RANGE; x_idx++) {
            for (int y_idx=0; y_idx < Y_RANGE; y_idx++) {
                assert(outData[x_idx][y_idx][z_idx] == inputData[z_idx][x_idx][y_idx]);
            }
        }
    }
}

int write_float_3D_data_log(float (*all_rv_matrices)[X_RANGE][Y_RANGE], const char* output_path, int z, int x, int y) {
    FILE *fp = fopen(output_path, "w+");
    if (NULL == fp) {
        fprintf(stderr, "fopen %s failed: %d\n", output_path, strerror(errno));
        return -1;
    }
 
    for (int h=0; h<z; h++) {
        for (int col=0; col<y; col++) {
            for (int row=0; row<x; row++) {
                int ret = fprintf(fp, "%.17f%c", all_rv_matrices[h][row][col], (row == 60)? '\n':' ');
                if (ret<0) {
                    fprintf(stderr, "write frame row %d failed: %d\n", row, strerror(errno));
                    fclose(fp);
                    return -1;
                }
            }
        }
    }
    fflush(fp);
    fclose(fp);
    printf("All 25 frames processed and appended to %s\n", output_path);
    return 0;
}

void osCfarDetectMmw3D(const float (*inputData)[X_RANGE][Y_RANGE], CFARConfig* config_zy, CFARConfig* config_zx, CFARConfig* config_xy, int frame, uint16_t (*bitMap)[X_RANGE][Y_RANGE]) {
    struct timespec start1, end1;

    bool (* bitMap_zx)[X_RANGE][Y_RANGE] = (bool (*)[X_RANGE][Y_RANGE])mkl_malloc(Z_RANGE*X_RANGE*Y_RANGE * sizeof(bool), 64);
    bool (* bitMap_zy)[Y_RANGE][X_RANGE] = (bool (*)[Y_RANGE][X_RANGE])mkl_malloc(Z_RANGE*Y_RANGE*X_RANGE * sizeof(bool), 64);
    bool (* bitMap_xy)[Y_RANGE][Z_RANGE] = (bool (*)[Y_RANGE][Z_RANGE])mkl_malloc(X_RANGE*Y_RANGE*Z_RANGE * sizeof(bool), 64);
    memset(bitMap_zx, 0, Z_RANGE * X_RANGE * Y_RANGE * sizeof(bool));
    memset(bitMap_zy, 0, Z_RANGE * Y_RANGE * X_RANGE * sizeof(bool));
    memset(bitMap_xy, 0, X_RANGE * Y_RANGE * Z_RANGE * sizeof(bool));

    float (* f_result_zy)[Y_RANGE][X_RANGE] = (float (*)[Y_RANGE][X_RANGE])mkl_malloc(Z_RANGE*Y_RANGE*X_RANGE * sizeof(float), 64);
    float (* f_result_xy)[Y_RANGE][Z_RANGE] = (float (*)[Y_RANGE][Z_RANGE])mkl_malloc(X_RANGE*Y_RANGE*Z_RANGE * sizeof(float), 64);
    
    if (NULL == inputData || NULL == config_zy || NULL == bitMap_zx || NULL == bitMap_zy || NULL == bitMap_xy || NULL == f_result_zy || NULL == f_result_xy) {
        fprintf(stderr, "osCfarDetectMmw3D: malloc or input NULL!\n");
        goto exit_free;
    }

    /*char save_path[100]="";
    snprintf(save_path, 100, "/home/lg/cfar/bin_to_rv/txtFile/result/r_log%d.txt", frame+1);
    write_float_3D_data_log(inputData, save_path, 31, 61, 51);*/

    for (int z_idx=0; z_idx < Z_RANGE; z_idx++) {
        for (int x_idx=0; x_idx < X_RANGE; x_idx++) {
            const float *srcRow = inputData[z_idx][x_idx];
            bool *bitMapData = bitMap_zx[z_idx][x_idx];
            cfarDetect1d(srcRow, Y_RANGE, config_zx, bitMapData);
        }
    }

    for (int z_idx=0; z_idx < Z_RANGE; z_idx++) {
        mat_transpose_avx2((const float*)inputData[z_idx], (float*)f_result_zy[z_idx], X_RANGE, Y_RANGE);
        assertMatTransfer(inputData, f_result_zy, z_idx);
        for (int y_idx=0; y_idx < Y_RANGE; y_idx++) {
            const float *srcRow = f_result_zy[z_idx][y_idx];
            bool *bitMapData = bitMap_zy[z_idx][y_idx];
            cfarDetect1d(srcRow, X_RANGE, config_zy, bitMapData);
        }
    }

    mat_zxy_transpose_xyz((const float*)inputData, f_result_xy);
    assert_zxy_transpose_xyz(inputData, f_result_xy);
    for (int x_idx=0; x_idx < X_RANGE; x_idx++) {
        for (int y_idx=0; y_idx < Y_RANGE; y_idx++) {
            const float *srcRow = f_result_xy[x_idx][y_idx];
            bool *bitMapData = bitMap_xy[x_idx][y_idx];
            cfarDetect1d(srcRow, Z_RANGE, config_xy, bitMapData);
        }
    }
    int count = 0;
    for (int z_idx=0; z_idx<Z_RANGE; z_idx++) {
        for (int x_idx=0; x_idx<X_RANGE; x_idx++) {
            for (int y_idx=0; y_idx<Y_RANGE; y_idx++) {
                bitMap[z_idx][x_idx][y_idx] = 0;
                if (bitMap_zy[z_idx][y_idx][x_idx] || bitMap_xy[x_idx][y_idx][z_idx] || bitMap_zx[z_idx][x_idx][y_idx]) { // || bitMap_zy[z_idx][y_idx][x_idx] || bitMap_xy[x_idx][y_idx][z_idx] || bitMap_zx[z_idx][x_idx][y_idx]
                    bitMap[z_idx][x_idx][y_idx] = 1;
                    count++;
                }
            }
        }
    }
    printf("os_cfar:num=%d\n", count);
    
exit_free:
    // ===================== 必须释放内存 =====================
    mkl_free(bitMap_zx);
    mkl_free(bitMap_zy);
    mkl_free(bitMap_xy);
    mkl_free(f_result_zy);
    mkl_free(f_result_xy);
}


void targetDetection() {
    uint16_t (*bitMap)[X_RANGE][Y_RANGE] = (uint16_t (*)[X_RANGE][Y_RANGE])malloc(sizeof(uint16_t)*TOTAL_POINTS);
    struct timespec start1, end1;
        
    CFARConfig config_zy = {.CFAR_AVG_LEFT = 20, .CFAR_AVG_RIGHT=20, .CFAR_GUARD_INT=4, .CFAR_OS_KVALUE=19, .CFAR_THR_KSCALE=0.53, .CFAR_THR_SCALE=-1.53};
    CFARConfig config_zx = {.CFAR_AVG_LEFT = 16, .CFAR_AVG_RIGHT=16, .CFAR_GUARD_INT=4, .CFAR_OS_KVALUE=15, .CFAR_THR_KSCALE=0.53, .CFAR_THR_SCALE=-1.53};
    CFARConfig config_xy = {.CFAR_AVG_LEFT = 8,  .CFAR_AVG_RIGHT=8,  .CFAR_GUARD_INT=4, .CFAR_OS_KVALUE=7, .CFAR_THR_KSCALE=0.53, .CFAR_THR_SCALE=-1.53};
    
    char input_path[250]="";
    float (*inputData)[X_RANGE][Y_RANGE] = (float (*)[X_RANGE][Y_RANGE])malloc(sizeof(float)*Z_RANGE*X_RANGE*Y_RANGE);
    snprintf(input_path, 250, "/home/lg/multi_station_probability_fusion/spaceFusionVersion_3D/spatialData/frame_%d_spatial.txt", frameId);
    read_float_3D_data(input_path, inputData);
    
    osCfarDetectMmw3D(inputData, &config_zy, &config_zx, &config_xy, frameId, bitMap);

    /*char save_path[100]="";
    snprintf(save_path, 100, "/home/lg/cfar/bin_to_rv/txtFile/result/3D_osCfar_result%d.txt", frameId);
    write_uint16_data(bitMap, save_path, frameId, Z_RANGE, X_RANGE, Y_RANGE);*/
    free(bitMap);
    free(*inputData);
}