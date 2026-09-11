#include <assert.h>
#include <errno.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifndef X_RANGE
#define X_RANGE 61
#endif

#ifndef Y_RANGE
#define Y_RANGE 51
#endif

#ifndef Z_RANGE
#define Z_RANGE 31
#endif

#define TOTAL_POINTS (X_RANGE * Y_RANGE * Z_RANGE)
#define CFAR_AVG_OFFSET 3.0f //门限

#ifndef ITERATIONS
#define ITERATIONS 1 
#endif

#ifndef CFAR_ENABLE_STATS
#define CFAR_ENABLE_STATS 1
#endif

typedef struct {
    int len;
    int CFAR_AVG_LEFT;
    int CFAR_AVG_RIGHT;
    int CFAR_GUARD_INT;
    int CFAR_OS_KVALUE;
    float CFAR_THR_KSCALE;
    float CFAR_THR_SCALE;
} CFARConfig;
/*
len: 数据长度
CFAR_AVG_LEFT: 左侧平均窗口长度
CFAR_AVG_RIGHT: 右侧平均窗口长度
CFAR_GUARD_INT: 守卫单元长度
CFAR_OS_KVALUE: OS-CFAR的K值
CFAR_THR_KSCALE: OS-CFAR的门限缩放因子
CFAR_THR_SCALE: OS-CFAR的门限缩放因子
*/

typedef struct {
    float* h_scratch;
    float* col_sums;
    float* col_max;
    float* col_avg_limit;
    float* col_peak_limit;
    int h_stride;
    int col_stride;
} CFARWorkspace;
/*
h_scratch: 临时缓冲区
col_sums: 列求和
col_max: 列最大值
col_avg_limit: 列平均门限
col_peak_limit: 列峰值门限
h_stride: 水平步长
col_stride: 列步长
*/

//向上对齐
static inline int round_up_16(int x) {
    return (x + 15) & ~15;
}

//循环边界处理
static inline int wrap_once(int idx, int n) {
    if (idx < 0) return idx + n;
    if (idx >= n) return idx - n;
    return idx;
}

// 用 AVX512 快速复制 n 个 float
static inline void copy_float_avx512(float* restrict dst,
                                     const float* restrict src,
                                     int n) 
/*
dst: 目标数组
src: 源数组
n: 复制的元素数量
*/
{
    int i = 0;
    for (; i <= n - 16; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_loadu_ps(src + i));
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

//把列统计信息清空
static inline void init_column_stats(CFARWorkspace* ws, int cols) {
    const __m512 v_zero = _mm512_setzero_ps();
    const __m512 v_min = _mm512_set1_ps(-1e30f);

    int c = 0;
    for (; c <= cols - 16; c += 16) {
        _mm512_store_ps(ws->col_sums + c, v_zero);
        _mm512_store_ps(ws->col_max + c, v_min);
    }
    for (; c < cols; c++) {
        ws->col_sums[c] = 0.0f;
        ws->col_max[c] = -1e30f;
    }
}

// X方向的第一层 Gate
static inline void finalize_column_limits(CFARWorkspace* ws,
                                          const CFARConfig* cfg_zx, const CFARConfig* cfg_zy) {
    /*
    cfg_zy对应x方向配置，cfg_zx对应y方向配置
    */
    const int rows = cfg_zy->len;
    const int cols = cfg_zx->len;
    const __m512 v_inv_rows = _mm512_set1_ps(1.0f / (float)rows);
    const __m512 v_avg_offset = _mm512_set1_ps(CFAR_AVG_OFFSET);
    const __m512 v_peak_scale = _mm512_set1_ps(cfg_zy->CFAR_THR_SCALE);

    int c = 0;
    for (; c <= cols - 16; c += 16) {
        __m512 sums = _mm512_load_ps(ws->col_sums + c);
        __m512 maxv = _mm512_load_ps(ws->col_max + c);
        _mm512_store_ps(ws->col_avg_limit + c,
                        _mm512_add_ps(_mm512_mul_ps(sums, v_inv_rows), v_avg_offset));
        _mm512_store_ps(ws->col_peak_limit + c, _mm512_add_ps(maxv, v_peak_scale));
    }
    for (; c < cols; c++) {
        ws->col_avg_limit[c] = ws->col_sums[c] / (float)rows + CFAR_AVG_OFFSET;
        ws->col_peak_limit[c] = ws->col_max[c] + cfg_zy->CFAR_THR_SCALE;
    }
}

//padding，求和，求最值
static inline void pad_row_circular_stats_ps(const float* restrict row,
                                             float* restrict padded,
                                             CFARWorkspace* ws,
                                             const CFARConfig* cfg,
                                             float* out_max,
                                             float* out_avg) 
/*
row: 输入的行数据
padded: 输出的带有padding的行数据
ws: CFAR工作空间
cfg: CFAR配置参数
out_max: 输出的最大值
out_avg: 输出的平均值
*/
{
    const int cols = cfg->len;
    const int total_h = cfg->CFAR_AVG_LEFT + cfg->CFAR_GUARD_INT;//保护单元和检测单元

    copy_float_avx512(padded, row + cols - total_h, total_h);//快速复制

    __m512 v_max = _mm512_set1_ps(-1e30f);
    __m512 v_sum = _mm512_setzero_ps();
    int c = 0;
    for (; c <= cols - 16; c += 16) {
        __m512 v = _mm512_loadu_ps(row + c);
        _mm512_storeu_ps(padded + total_h + c, v);

        v_max = _mm512_max_ps(v_max, v);
        v_sum = _mm512_add_ps(v_sum, v);

        __m512 col_sum = _mm512_load_ps(ws->col_sums + c);
        __m512 col_max = _mm512_load_ps(ws->col_max + c);
        _mm512_store_ps(ws->col_sums + c, _mm512_add_ps(col_sum, v));
        _mm512_store_ps(ws->col_max + c, _mm512_max_ps(col_max, v));
    }

    float r_max = _mm512_reduce_max_ps(v_max);
    float r_sum = _mm512_reduce_add_ps(v_sum);

    for (; c < cols; c++) {
        float v = row[c];
        padded[total_h + c] = v;
        if (v > r_max) r_max = v;
        r_sum += v;
        ws->col_sums[c] += v;
        if (v > ws->col_max[c]) ws->col_max[c] = v;
    }

    copy_float_avx512(padded + total_h + cols, row, total_h);

    *out_max = r_max;
    *out_avg = r_sum / (float)cols;
}

//把 AVX512 一次处理出的16个布尔结果写回 bitmap
static inline void store_vertical_mask_bool(bool (*dst)[X_RANGE],
                                            int row,
                                            int col,
                                            __mmask16 mask)
/*
dst: 输出的布尔矩阵
row: 当前行索引
col: 当前列索引
mask: 输出结果
*/
 {
    for (int lane = 0; lane < 16; lane++) {
        dst[col + lane][row] = (mask & (1u << lane)) ? true : false;
    }
}

//连续方向的一维 OS-CFAR 核心
static void os_cfar_avx512(const float* padded_row,
                                      const float* cut_row,
                                      bool* results,
                                      const CFARConfig* cfg,
                                      float avg_limit,
                                      float peak_limit) 
/*
padded_row: 带有padding的行数据
cut_row: 当前行的CUT数据
results: 输出的布尔结果
cfg: CFAR配置参数
avg_limit: 平均门限
peak_limit: 峰值门限
*/
{
    const int cols = cfg->len;
    const int guard = cfg->CFAR_GUARD_INT;
    const int total_h = cfg->CFAR_AVG_LEFT + guard;
    const int k_rank = cfg->CFAR_OS_KVALUE;
    const float offset = cfg->CFAR_THR_KSCALE;

    const __m512 v_offset = _mm512_set1_ps(offset);
    const __m512 v_avg_limit = _mm512_set1_ps(avg_limit);
    const __m512 v_peak_limit = _mm512_set1_ps(peak_limit);
    const __m512i v_one = _mm512_set1_epi32(1);
    const __m512i v_k = _mm512_set1_epi32(k_rank );

    int i = 0;
    //处理分块矩阵
    for (; i <= cols - 16; i += 16) 
    {
        __m512 v_cuts = _mm512_loadu_ps(cut_row + i);//加载16个CUT

        __mmask16 m_gate =  _mm512_cmp_ps_mask(v_cuts, v_avg_limit, _CMP_GE_OQ) ;//平均值gate
        // __mmask16 m_gate =  
        //     _mm512_cmp_ps_mask(v_cuts, v_avg_limit, _CMP_GE_OQ) &
        //     _mm512_cmp_ps_mask(v_cuts, v_peak_limit, _CMP_GE_OQ);
        if (m_gate == 0) {
            continue;
        }

        //局部最大值
        __m512 v_left = _mm512_loadu_ps(padded_row + total_h + i - 1);
        __m512 v_right = _mm512_loadu_ps(padded_row + total_h + i + 1);
        __mmask16 m_max =
            _mm512_cmp_ps_mask(v_cuts, v_left, _CMP_GE_OQ) &
            _mm512_cmp_ps_mask(v_cuts, v_right, _CMP_GE_OQ);
        if (m_max == 0) {
            continue;
        }
        
        //OS-CFAR：计算每个CUT的背景噪声估计值，并与CUT进行比较
        __m512 v_thr = _mm512_sub_ps(v_cuts, v_offset);
        __m512i cnt0 = _mm512_setzero_si512();
        __m512i cnt1 = _mm512_setzero_si512();
        __m512i cnt2 = _mm512_setzero_si512();
        __m512i cnt3 = _mm512_setzero_si512();

        //遍历
        int bucket = 0;
        /*
        v_ref : 参考单元的值
        v_thr : 当前CUT的阈值
        mask : 参考单元是否小于等于阈值的掩码
        */
        for (int off = -total_h; off < -guard; off++) {
            __m512 v_ref = _mm512_loadu_ps(padded_row + i + total_h + off);
            __mmask16 mask = _mm512_cmp_ps_mask(v_ref, v_thr, _CMP_LE_OQ);
            switch (bucket & 3) {
                case 0: cnt0 = _mm512_mask_add_epi32(cnt0, mask, cnt0, v_one); break;
                case 1: cnt1 = _mm512_mask_add_epi32(cnt1, mask, cnt1, v_one); break;
                case 2: cnt2 = _mm512_mask_add_epi32(cnt2, mask, cnt2, v_one); break;
                default: cnt3 = _mm512_mask_add_epi32(cnt3, mask, cnt3, v_one); break;
            }
            bucket++;//计数器
        }

        for (int off = guard + 1; off <= total_h; off++) {
            __m512 v_ref = _mm512_loadu_ps(padded_row + i + total_h + off);
            __mmask16 mask = _mm512_cmp_ps_mask(v_ref, v_thr, _CMP_LE_OQ);
            switch (bucket & 3) {
                case 0: cnt0 = _mm512_mask_add_epi32(cnt0, mask, cnt0, v_one); break;
                case 1: cnt1 = _mm512_mask_add_epi32(cnt1, mask, cnt1, v_one); break;
                case 2: cnt2 = _mm512_mask_add_epi32(cnt2, mask, cnt2, v_one); break;
                default: cnt3 = _mm512_mask_add_epi32(cnt3, mask, cnt3, v_one); break;
            }
            bucket++;
        }
        //合并计数器（四个计数器同时工作）
        __m512i counts = _mm512_add_epi32(_mm512_add_epi32(cnt0, cnt1),
                                          _mm512_add_epi32(cnt2, cnt3));
        /*
        counts: 每个CUT的小于判决的计数
        */

        //k判决
        __mmask16 m_rank = _mm512_cmpgt_epi32_mask(counts, v_k);


        for (int lane = 0; lane < 16; lane++) {
            results[i + lane] = (m_rank & m_max & m_gate) & (1u << lane) ? true : false;
        }
    }


    //处理剩余不足16的列
    for (; i < cols; i++) {
        float cur = cut_row[i];
        // int pass_gate = (cur >= avg_limit) && (cur >= peak_limit);
        int pass_gate = (cur >= avg_limit) ;
        if (pass_gate &&
            cur >= padded_row[total_h + i - 1] &&
            cur >= padded_row[total_h + i + 1]) {
            float thr = cur - offset;
            int count = 0;
            for (int off = -total_h; off < -guard; off++) {
                if (padded_row[i + total_h + off] <= thr) count++;
            }
            for (int off = guard + 1; off <= total_h; off++) {
                if (padded_row[i + total_h + off] <= thr) count++;
            }
            results[i] = count > k_rank;
        } else {
            results[i] = false;
        }
    }
}

//纵向CFAR,不需要padding，直接在原矩阵上计算，采用索引访问，对y方向做处理，矩阵构造为x行y列
static void vertical_cfar_direct_avx512(const float* matrix,
                                    bool (*bitMap_zy)[X_RANGE],
                                    const CFARConfig* cfg_zx,
                                    const CFARConfig* cfg_zy,
                                    const CFARWorkspace* ws)
/*
matrix: 输入的二维数据矩阵
bitMap_zy: 输出的二值化检测结果矩阵
cfg_zx: x方向的CFAR配置参数
cfg_zy: y方向的CFAR配置参数
ws: CFAR工作空间
*/
{
    const int rows = cfg_zy->len;
    const int cols = cfg_zx->len;
    const int guard = cfg_zy->CFAR_GUARD_INT;
    const int total_h = cfg_zy->CFAR_AVG_LEFT + guard;
    const int k_rank = cfg_zy->CFAR_OS_KVALUE;
    const float offset = cfg_zy->CFAR_THR_KSCALE;

    const __m512 v_offset = _mm512_set1_ps(offset);
    const __m512i v_one = _mm512_set1_epi32(1);
    const __m512i v_k = _mm512_set1_epi32(k_rank );

    const int vec_cols = (cols / 16) * 16;

    //处理分块矩阵
    for (int c = 0; c < vec_cols; c += 16) 
    {
        __m512 v_avg_limit = _mm512_load_ps(ws->col_avg_limit + c);
        __m512 v_peak_limit = _mm512_load_ps(ws->col_peak_limit + c);

        for (int r = 0; r < rows; r++) //遍历x方向
        {
            const float* cut_ptr = matrix + r * cols + c;
            /*
            cut_ptr: 指向当前行的指针
            v_cuts: 当前行的16个CUT值
            */
            __m512 v_cuts = _mm512_loadu_ps(cut_ptr);

            // __mmask16 m_gate =
            //     _mm512_cmp_ps_mask(v_cuts, v_avg_limit, _CMP_GE_OQ) &
            //     _mm512_cmp_ps_mask(v_cuts, v_peak_limit, _CMP_GE_OQ);
            __mmask16 m_gate =
                _mm512_cmp_ps_mask(v_cuts, v_avg_limit, _CMP_GE_OQ);//平均门限判断
            if (m_gate == 0) continue;

            //局部最大值判断
            int prev_r = (r == 0) ? rows - 1 : r - 1;
            int next_r = (r + 1 == rows) ? 0 : r + 1;
            __m512 v_prev = _mm512_loadu_ps(matrix + prev_r * cols + c);
            __m512 v_next = _mm512_loadu_ps(matrix + next_r * cols + c);
            __mmask16 m_max =
                _mm512_cmp_ps_mask(v_cuts, v_prev, _CMP_GE_OQ) &
                _mm512_cmp_ps_mask(v_cuts, v_next, _CMP_GE_OQ);
            if (m_max == 0) continue;

            //判决门限计算
            __m512 v_thr = _mm512_sub_ps(v_cuts, v_offset);
            __m512i cnt0 = _mm512_setzero_si512();
            __m512i cnt1 = _mm512_setzero_si512();
            __m512i cnt2 = _mm512_setzero_si512();
            __m512i cnt3 = _mm512_setzero_si512();

            int bucket = 0;
            for (int off = -total_h; off < -guard; off++) {
                int rr = wrap_once(r + off, rows);
                __m512 v_ref = _mm512_loadu_ps(matrix + rr * cols + c);
                __mmask16 mask = _mm512_cmp_ps_mask(v_ref, v_thr, _CMP_LE_OQ);
                switch (bucket & 3) {
                    case 0: cnt0 = _mm512_mask_add_epi32(cnt0, mask, cnt0, v_one); break;
                    case 1: cnt1 = _mm512_mask_add_epi32(cnt1, mask, cnt1, v_one); break;
                    case 2: cnt2 = _mm512_mask_add_epi32(cnt2, mask, cnt2, v_one); break;
                    default: cnt3 = _mm512_mask_add_epi32(cnt3, mask, cnt3, v_one); break;
                }
                bucket++;
            }

            for (int off = guard + 1; off <= total_h; off++) 
            //遍历上参考行
            /*
            rr: 当前参考行
            v_ref: 当前*参考行*的16个值
            v_thr: 当前CUT的阈值
            */
            {
                int rr = wrap_once(r + off, rows);//当前要读取的参考行号
                __m512 v_ref = _mm512_loadu_ps(matrix + rr * cols + c);//加载matrix[rr][c]连续16位
                __mmask16 mask = _mm512_cmp_ps_mask(v_ref, v_thr, _CMP_LE_OQ);//比较参考行的值是否小于等于阈值
                switch (bucket & 3) {
                    case 0: cnt0 = _mm512_mask_add_epi32(cnt0, mask, cnt0, v_one); break;
                    case 1: cnt1 = _mm512_mask_add_epi32(cnt1, mask, cnt1, v_one); break;
                    case 2: cnt2 = _mm512_mask_add_epi32(cnt2, mask, cnt2, v_one); break;
                    default: cnt3 = _mm512_mask_add_epi32(cnt3, mask, cnt3, v_one); break;
                }
                bucket++;
            }

            __m512i counts = _mm512_add_epi32(_mm512_add_epi32(cnt0, cnt1),
                                              _mm512_add_epi32(cnt2, cnt3));
            __mmask16 m_rank = _mm512_cmpgt_epi32_mask(counts, v_k);
            __mmask16 m_final = m_rank & m_max & m_gate;
            if (m_final != 0) {
                store_vertical_mask_bool(bitMap_zy, r, c, m_final);
            }
        }
    }
    //标量代码完成不足16的列
    for (int c = vec_cols; c < cols; c++) {
        float avg_limit = ws->col_avg_limit[c];
        float peak_limit = ws->col_peak_limit[c];

        for (int r = 0; r < rows; r++) {
            float cur = matrix[r * cols + c];
            // int pass_gate = (cur >= avg_limit) && (cur >= peak_limit);
            int pass_gate = (cur >= avg_limit);
            if (!pass_gate) continue;

            int prev_r = (r == 0) ? rows - 1 : r - 1;
            int next_r = (r + 1 == rows) ? 0 : r + 1;
            if (cur < matrix[prev_r * cols + c] ||
                cur < matrix[next_r * cols + c]) {
                continue;
            }

            float thr = cur - offset;
            int count = 0;
            for (int off = -total_h; off < -guard; off++) {
                int rr = wrap_once(r + off, rows);
                if (matrix[rr * cols + c] <= thr) count++;
            }
            for (int off = guard + 1; off <= total_h; off++) {
                int rr = wrap_once(r + off, rows);
                if (matrix[rr * cols + c] <= thr) count++;
            }
            if (count > k_rank) {
                bitMap_zy[c][r] = true;
            }
        }
    }
}

//创建CFAR临时工作空间
static CFARWorkspace* init_workspace(const CFARConfig* cfg) 
/*
cfg: CFAR配置参数
ws: 返回的CFAR工作空间
*/
{
    int total_h = cfg->CFAR_AVG_LEFT + cfg->CFAR_GUARD_INT;
    CFARWorkspace* ws = (CFARWorkspace*)malloc(sizeof(CFARWorkspace));
    if (ws == NULL) return NULL;

    ws->h_stride = round_up_16(cfg->len + 2 * total_h );//实际需要的长度
    ws->col_stride = round_up_16(cfg->len);

    ws->h_scratch = (float*)_mm_malloc((size_t)ws->h_stride * sizeof(float),64);
    ws->col_sums = (float*)_mm_malloc((size_t)ws->col_stride * sizeof(float),64);
    ws->col_max = (float*)_mm_malloc((size_t)ws->col_stride * sizeof(float),64);
    ws->col_avg_limit = (float*)_mm_malloc((size_t)ws->col_stride * sizeof(float),64);
    ws->col_peak_limit = (float*)_mm_malloc((size_t)ws->col_stride * sizeof(float),64);

    if (!ws->h_scratch || !ws->col_sums || !ws->col_max ||
        !ws->col_avg_limit || !ws->col_peak_limit) {
        _mm_free(ws->h_scratch);
        _mm_free(ws->col_sums);
        _mm_free(ws->col_max);
        _mm_free(ws->col_avg_limit);
        _mm_free(ws->col_peak_limit);
        free(ws);
        return NULL;
    }

    return ws;
}

//释放内存
static void free_workspace(CFARWorkspace* ws) {
    if (ws == NULL) return;
    _mm_free(ws->h_scratch);
    _mm_free(ws->col_sums);
    _mm_free(ws->col_max);
    _mm_free(ws->col_avg_limit);
    _mm_free(ws->col_peak_limit);
    free(ws);
}

//检查CFAR配置参数是否有效
static int valid_config_for_len(const CFARConfig* cfg, int len) {
    if (cfg == NULL) return 0;
    if (cfg->CFAR_AVG_LEFT <= 0 || cfg->CFAR_AVG_RIGHT <= 0) return 0;
    if (cfg->CFAR_GUARD_INT < 0 || cfg->CFAR_OS_KVALUE <= 0) return 0;
    if (cfg->CFAR_OS_KVALUE > cfg->CFAR_AVG_LEFT + cfg->CFAR_AVG_RIGHT) return 0;
    if (cfg->CFAR_AVG_LEFT + cfg->CFAR_GUARD_INT >= len) return 0;
    if (cfg->CFAR_AVG_RIGHT + cfg->CFAR_GUARD_INT >= len) return 0;
    return 1;
}


//固定z轴的二维 CFAR 总调度函数
static void cfar_process_ps(const float (*input)[Y_RANGE],
                                  bool (*bitMap_zx)[Y_RANGE],
                                  bool (*bitMap_zy)[X_RANGE],
                                  const CFARConfig* cfg_zx,
                                  const CFARConfig* cfg_zy,
                                  CFARWorkspace* ws) 
/*
input: 输入的二维数据矩阵
bitMap_zx: 输出的X方向二值化检测结果矩阵
bitMap_zy: 输出的Y方向二值化检测结果矩阵
cfg_zx: X方向的CFAR配置参数
cfg_zy: Y方向的CFAR配置参数
ws: CFAR工作空间
*/
{
    memset(bitMap_zx, 0, (size_t)X_RANGE * Y_RANGE * sizeof(bool));
    memset(bitMap_zy, 0, (size_t)X_RANGE * Y_RANGE * sizeof(bool));
    init_column_stats(ws, cfg_zx->len);

    for (int x = 0; x < cfg_zy->len; x++) {
        float row_max, row_avg;
        pad_row_circular_stats_ps(input[x], ws->h_scratch, ws, cfg_zx,
                                  &row_max, &row_avg);
        //对x方向做处理
        os_cfar_avx512(ws->h_scratch, input[x], bitMap_zx[x],
                                  cfg_zx,
                                  row_avg + CFAR_AVG_OFFSET,
                                  row_max + cfg_zx->CFAR_THR_SCALE);
    }

    finalize_column_limits(ws, cfg_zx, cfg_zy);
    vertical_cfar_direct_avx512((const float*)input, bitMap_zy, cfg_zx, cfg_zy, ws);
}

//封装
void zhwx_oscfar(const float (*input)[Y_RANGE],
                 bool (*bitMap_zx)[Y_RANGE],
                 bool (*bitMap_zy)[X_RANGE],
                 const CFARConfig* config_zx,
                 const CFARConfig* config_zy) {

    CFARWorkspace* ws = init_workspace(config_zx);
    if (ws == NULL) return;

    cfar_process_ps(input, bitMap_zx, bitMap_zy, config_zx, config_zy, ws);
    free_workspace(ws);
}

//把zxy矩阵转置为xyz矩阵
static void transpose_zxy_to_xyz(const float (*input)[X_RANGE][Y_RANGE],
                                 float (*out)[Y_RANGE][Z_RANGE]) {
    for (int z = 0; z < Z_RANGE; z++) {
        for (int x = 0; x < X_RANGE; x++) {
            for (int y = 0; y < Y_RANGE; y++) {
                out[x][y][z] = input[z][x][y];
            }
        }
    }
}


void osCfarDetectMmw3D(const float (*inputData)[X_RANGE][Y_RANGE],
                       CFARConfig* config_zy,
                       CFARConfig* config_zx,
                       CFARConfig* config_xy,
                       int frame,
                       uint16_t (*bitMap)[X_RANGE][Y_RANGE]) 
/*
inputData: 输入的三维数据矩阵
config_zy: Y方向的CFAR配置参数
config_zx: X方向的CFAR配置参数
config_xy: Z方向的CFAR配置参数
frame: 当前帧号
bitMap: 输出的三维二值化检测结果矩阵
*/
{
    //申请bitmap
    bool (*bitMap_zx)[X_RANGE][Y_RANGE] =
        (bool (*)[X_RANGE][Y_RANGE])_mm_malloc((size_t)Z_RANGE * X_RANGE * Y_RANGE * sizeof(bool), 64);
    bool (*bitMap_zy)[Y_RANGE][X_RANGE] =
        (bool (*)[Y_RANGE][X_RANGE])_mm_malloc((size_t)Z_RANGE * Y_RANGE * X_RANGE * sizeof(bool), 64);
    bool (*bitMap_xy)[Y_RANGE][Z_RANGE] =
        (bool (*)[Y_RANGE][Z_RANGE])_mm_malloc((size_t)X_RANGE * Y_RANGE * Z_RANGE * sizeof(bool), 64);


    // float (*f_result)[X_RANGE][Y_RANGE] =
    //     (float (*)[X_RANGE][Y_RANGE])_mm_malloc((size_t)Z_RANGE * X_RANGE * Y_RANGE * sizeof(float), 64);
    const float (*f_result)[X_RANGE][Y_RANGE] = inputData;

    float (*f_result_xy)[Y_RANGE][Z_RANGE] =
        (float (*)[Y_RANGE][Z_RANGE])_mm_malloc((size_t)X_RANGE * Y_RANGE * Z_RANGE * sizeof(float), 64);

    if (inputData == NULL || config_zy == NULL || config_zx == NULL ||
        config_xy == NULL || bitMap == NULL ||
        bitMap_zx == NULL || bitMap_zy == NULL || bitMap_xy == NULL ||
        f_result == NULL || f_result_xy == NULL) {
        fprintf(stderr, "osCfarDetectMmw3D: allocation or input failed\n");
        goto exit_free;
    }

    memset(bitMap_zx, 0, (size_t)Z_RANGE * X_RANGE * Y_RANGE * sizeof(bool));
    memset(bitMap_zy, 0, (size_t)Z_RANGE * Y_RANGE * X_RANGE * sizeof(bool));
    memset(bitMap_xy, 0, (size_t)X_RANGE * Y_RANGE * Z_RANGE * sizeof(bool));

    // const float* src = (const float*)inputData;
    // float* dst = (float*)f_result;

    // //可以用mkl库替换
    // // for (int i = 0; i < TOTAL_POINTS; i++) {
    // //     dst[i] = log2f(src[i]);
    // // }
    // for (int i = 0; i < TOTAL_POINTS; i++) {
    //     dst[i] = src[i];
    // } 

    //针对每一个z,做二维CFAR
    for (int z = 0; z < Z_RANGE; z++) {
        zhwx_oscfar(f_result[z], bitMap_zx[z], bitMap_zy[z], config_zx, config_zy);
    }

    // float *h_scratch = (float*)_mm_malloc((size_t)(Z_RANGE + 2 * (config_xy->CFAR_AVG_LEFT + config_xy->CFAR_GUARD_INT)) * sizeof(float),64);
    // if(h_scratch == NULL) {
    //     fprintf(stderr, "Failed to allocate scratch memory\n");
    //     goto exit_free;
    // }
    CFARWorkspace* ws_z = init_workspace(config_xy);
    if (ws_z == NULL) 
    {
        printf("Failed to initialize workspace ws_z\n");
        goto exit_free;
    }
    init_column_stats(ws_z, config_xy->len);

    transpose_zxy_to_xyz(f_result, f_result_xy);
    for (int x = 0; x < X_RANGE; x++) {
        for (int y = 0; y < Y_RANGE; y++) {
            // cfar_detect_1d(f_result_xy[x][y], Z_RANGE, config_xy, bitMap_xy[x][y]);

            float row_max, row_avg;
            pad_row_circular_stats_ps(f_result_xy[x][y], ws_z->h_scratch, ws_z, config_xy,
                                    &row_max, &row_avg);

            os_cfar_avx512(ws_z->h_scratch,f_result_xy[x][y], bitMap_xy[x][y], config_xy,
                            row_avg + CFAR_AVG_OFFSET,
                            row_max + config_xy->CFAR_THR_SCALE);
        }
    }
    free_workspace(ws_z);
    //三方向OR
    for (int z = 0; z < Z_RANGE; z++) {
        for (int x = 0; x < X_RANGE; x++) {
            for (int y = 0; y < Y_RANGE; y++) {
                bitMap[z][x][y] =
                    (bitMap_zx[z][x][y] ||
                     bitMap_zy[z][y][x] ||
                     bitMap_xy[x][y][z]) ? 1u : 0u;
            }
        }
    }



exit_free:
    _mm_free(bitMap_zx);
    _mm_free(bitMap_zy);
    _mm_free(bitMap_xy);
    // _mm_free(f_result);
    _mm_free(f_result_xy);
}

//计时
static double timeval_diff_ms(struct timeval* start, struct timeval* end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_usec - start->tv_usec) / 1000.0;
}

//保存文件
static void save_bin(const char* path, const void* data, size_t size) {
    FILE* fp = fopen(path, "wb");
    if (fp) {
        fwrite(data, size, 1, fp);
        fclose(fp);
    }
}

//读取三维浮点数据
void read_float_3D_data(const char *filename, float (*data)[X_RANGE][Y_RANGE])
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("文件打开失败");
        printf("[DEBUG] 尝试打开文件: %s\n", filename);
        return;
    }


    for (int z = 0; z < Z_RANGE; z++) {
        for (int y = 0; y < Y_RANGE; y++) {
            for (int x = 0; x < X_RANGE; x++) {
                if (fscanf(fp, "%f", &data[z][x][y]) != 1) {
                    printf("读取失败 z=%d x=%d y=%d\n", z, x, y);
                    fclose(fp);
                    return;
                }
            }
        }
    }

    fclose(fp);
    printf("✅ %s, 读取完成：1891行 51列\n", filename);
}


int main(void) {
    int iter = ITERATIONS;

    float (*matrix)[X_RANGE][Y_RANGE] =
        (float (*)[X_RANGE][Y_RANGE])_mm_malloc((size_t)Z_RANGE * X_RANGE * Y_RANGE * sizeof(float), 64);
    uint16_t (*bitMap)[X_RANGE][Y_RANGE] =
        (uint16_t (*)[X_RANGE][Y_RANGE])_mm_malloc((size_t)Z_RANGE * X_RANGE * Y_RANGE * sizeof(uint16_t), 64);


    if (matrix == NULL || bitMap == NULL) {
        printf("allocation failed\n");
        _mm_free(matrix);
        _mm_free(bitMap);
        return -1;
    }

    CFARConfig config_zy = {.len = X_RANGE, .CFAR_AVG_LEFT = 20, .CFAR_AVG_RIGHT=20, .CFAR_GUARD_INT=4, .CFAR_OS_KVALUE=19, .CFAR_THR_KSCALE=0.53, .CFAR_THR_SCALE=-1.53};
    CFARConfig config_zx = {.len = Y_RANGE, .CFAR_AVG_LEFT = 16, .CFAR_AVG_RIGHT=16, .CFAR_GUARD_INT=4, .CFAR_OS_KVALUE=15, .CFAR_THR_KSCALE=0.53, .CFAR_THR_SCALE=-1.53};
    CFARConfig config_xy = {.len = Z_RANGE, .CFAR_AVG_LEFT = 8, .CFAR_AVG_RIGHT=8, .CFAR_GUARD_INT=4, .CFAR_OS_KVALUE=7, .CFAR_THR_KSCALE=0.53, .CFAR_THR_SCALE=-1.53};
    
    for (int frameId = 1; frameId <= 24; frameId++) {

        char input_path[250]="";
        snprintf(input_path, 250, "./data_trace6_log/r_log%d.txt", frameId);
        read_float_3D_data(input_path, matrix);


        printf("Starting OS-CFAR [%d x %d x %d]...\n",Z_RANGE, X_RANGE, Y_RANGE);

        struct timeval start, end;
        gettimeofday(&start, NULL);
        for (int n = 0; n < iter; n++) {
            osCfarDetectMmw3D(matrix, &config_zy, &config_zx, &config_xy, frameId, bitMap);
        }
        gettimeofday(&end, NULL);

        double t1 = timeval_diff_ms(&start, &end);
        printf("AVX512 Time (%d iter): %.2f ms\n", iter, t1);

        char out_file[256];
        sprintf(out_file, "./c_outputs/r_log%02d_det.bin", frameId);
        save_bin(out_file, bitMap, Z_RANGE * X_RANGE * Y_RANGE * sizeof(uint16_t));
        printf("\nData exported to .bin files for MATLAB.\n");
    }

    _mm_free(matrix);
    _mm_free(bitMap);
    return 0;
}
