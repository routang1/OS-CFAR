#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h> 
#include "commDef.h"
#include "osCfar.h"

// 交换两个double数（排序用）
void swap(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

// 选择排序
void selectionSort(double *arr, int len) {
    int i, j, min_idx;
    for (i = 0; i < len - 1; i++) {
        min_idx = i;
        for (j = i + 1; j < len; j++) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        swap(&arr[min_idx], &arr[i]);
    }
}

// 将数组循环右移 shift 位
void arrayRoll(const double *in, double *out, int len, int shift) {
    shift = (shift % len + len) % len; 
    for (int i = 0; i < len; i++) {
        int idx = (i - shift + len) % len;
        out[i] = in[idx];
    }
}

//局部最大判断
void localMaxDetect(const double *data, int len, bool *out) {
    if (NULL == data || NULL == out) {
        fprintf(stderr, "invalid input params");
        return;
    }

    double *shiftR1 = (double *)malloc(len * sizeof(double));
    double *shiftL1 = (double *)malloc(len * sizeof(double));
    if (shiftR1 == NULL || shiftL1 == NULL) {
        fprintf(stderr, "Memory allocation failed for shift buffers! (len=%d)\n", len);
        free(shiftR1);
        free(shiftL1);
        return;
    }
 
    arrayRoll(data, shiftR1, len, 1);
    arrayRoll(data, shiftL1, len, -1);

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

//估计CUT周围背景噪声
void osCfarNoiseFinder(const double *data, int col, CFARConfig* config, double *noiseOut) {
    if (NULL == data || NULL == config || NULL == noiseOut) {
       fprintf(stderr, "osCfarNoiseFinder exist input params is null");
       return;
    }

    int winLen = config->CFAR_AVG_LEFT+config->CFAR_AVG_RIGHT;
    double *wnBuffer = (double*)malloc(winLen*sizeof(double));
    if (NULL == wnBuffer) {
        fprintf(stderr, "Memory allocation failed for statWin! (statWinLen=%d)\n", winLen);
        return;
    }

    if (config->CFAR_OS_KVALUE >= winLen) {
        fprintf(stderr, "osCfarNoiseFinder: Invalid CFAR configuration\n");
        return;
    }

    int curLeftIdx=0;
    int curRightIdx=0;
    for (int i=0; i< col; i++) {
        memset(wnBuffer, 0, winLen*sizeof(double));
        for (int leftIdx=0; leftIdx<config->CFAR_AVG_LEFT; leftIdx++) {
            curLeftIdx = i - config->CFAR_AVG_LEFT - config->CFAR_GUARD_INT + leftIdx;
            if (curLeftIdx < 0) {
                curLeftIdx += col;
            }
            if (curLeftIdx >=0 ) wnBuffer[leftIdx] = data[curLeftIdx];//=0注意
        }

        for (int rightIdx=0; rightIdx<config->CFAR_AVG_RIGHT; rightIdx++) {
            curRightIdx = i + config->CFAR_GUARD_INT + rightIdx + 1;
            if (curRightIdx >= col) {
                curRightIdx -= col;
            }
            if ((config->CFAR_AVG_LEFT + rightIdx < winLen) && curRightIdx >= 0)
            wnBuffer[config->CFAR_AVG_LEFT + rightIdx] = data[curRightIdx];
        }
        selectionSort(wnBuffer, winLen);
        noiseOut[i] = wnBuffer[config->CFAR_OS_KVALUE];
    }
    free(wnBuffer);
}

//1D osCFAR
void osCfarDetectMmw1D(const double *dbData, int row, int col, CFARConfig* config, bool *bitMap, double *noiseMap) {
    if (NULL ==dbData || NULL == noiseMap || NULL == bitMap || NULL == config || 0 == row || 0 == col) {
        fprintf(stderr, "Invalid input params!\n");
        return;
    }

    bool *localMax = (bool*)malloc(col*sizeof(bool));
    if (NULL == localMax) {
        fprintf(stderr, "malloc localMax failed!\n");
        return;
    }

    for (int rowIdx=0; rowIdx<row; rowIdx++) {
        const double *srcRow = dbData + rowIdx*col;
        double *noiseMapRow = noiseMap + rowIdx*col;
        osCfarNoiseFinder(srcRow, col, config, noiseMapRow);//逐行检测背景噪声

        memset(localMax, 0, col*sizeof(bool));
        localMaxDetect(srcRow, col, localMax);

        for (int i=0; i<col; i++) {
            double threshold = noiseMapRow[i] + config->CFAR_DET_THR;
            bitMap[rowIdx*col+i] = (srcRow[i] >= threshold) && localMax[i];
        }
    }

    free(localMax);
}
//分块转置
#define BLOCK_SIZE 16 // 根据 CPU 缓存行调整，通常 16, 32, 64
//转置为row行col列
void matriTransponse(double *in, double *out, int row, int col) {
    for (int jj = 0; jj < col; jj += BLOCK_SIZE) {
        for (int ii = 0; ii < row; ii += BLOCK_SIZE) {
            for (int j = jj; j < jj + BLOCK_SIZE && j < col; j++) {
                for (int i = ii; i < ii + BLOCK_SIZE && i < row; i++) {
                    out[i*col+j] = in[j*row+i];
                }
            }
        }
    }
}


void boolMatriTransponse(bool *in, bool *out) {
    for (int jj = 0; jj < SCAN_CNT; jj += BLOCK_SIZE) {
        for (int ii = 0; ii < SAMPLE_CNT; ii += BLOCK_SIZE) {
            for (int j = jj; j < jj + BLOCK_SIZE && j < SCAN_CNT; j++) {
                for (int i = ii; i < ii + BLOCK_SIZE && i < SAMPLE_CNT; i++) {
                    out[i*SCAN_CNT+j] = in[j*SAMPLE_CNT+i];
                }
            }
        }
    }
}


// 2D OS-CFAR 检测
void osCfarDetectMmw2D(double *dbData, CFARConfig* config, double *bitMap, double *noiseMap) {
    /**
     *dbData: 输入的二维数据矩阵
     *config: CFAR配置参数
     *bitMap: 输出的二值化检测结果矩阵
     *noiseMap: 输出的背景噪声估计矩阵
     */
    
    //检测512行
    bool *bitMapA = (bool *)malloc(sizeof(bool)*SAMPLE_CNT*SCAN_CNT);
    bool *bitMapB = (bool *)malloc(sizeof(bool)*SCAN_CNT*SAMPLE_CNT);
    double *noiseMapA = (double*)malloc(sizeof(double)*SAMPLE_CNT*SCAN_CNT);
    double *noiseMapB = (double*)malloc(sizeof(double)*SCAN_CNT*SAMPLE_CNT);

    if (NULL == dbData || NULL == bitMapA || NULL == noiseMapA || NULL == bitMapB || NULL == noiseMapB || 
        NULL == bitMap || NULL == config) {
        fprintf(stderr, "osCfarDetectMmw2D: malloc failed!\n");
        free(bitMapA);
        free(bitMapB);
        free(noiseMapA);
        free(noiseMapB);
        return;
    }

    osCfarDetectMmw1D(dbData, SAMPLE_CNT, SCAN_CNT, config, bitMapA, noiseMapA);

    double *dbDataTransponse = (double*)malloc(sizeof(double)*SCAN_CNT*SAMPLE_CNT);
    if (NULL == dbDataTransponse) {
        free(dbDataTransponse);
        fprintf(stderr, "osCfarDetectMmw2D: malloc dbDataTransponse failed!\n");
        return;
    }
    matriTransponse(dbData, dbDataTransponse, SCAN_CNT, SAMPLE_CNT);//1024*512的double型矩阵专置为512*1024
    osCfarDetectMmw1D(dbDataTransponse, SCAN_CNT, SAMPLE_CNT, config, bitMapB, noiseMapB);//对每一行的1024列计算oscfar

    bool *bitMapBTrans = (bool*)malloc(sizeof(bool)*SAMPLE_CNT*SCAN_CNT);//1024*512
    double *noiseMapBTrans = (double*)malloc(sizeof(double)*SAMPLE_CNT*SCAN_CNT);//1024*512
    if (NULL == bitMapBTrans || NULL == noiseMapBTrans) {
        free(bitMapBTrans);
        fprintf(stderr, "osCfarDetectMmw2D: malloc bitMapBtrans failed!\n");
        return;
    }
    boolMatriTransponse(bitMapB, bitMapBTrans);
    matriTransponse(noiseMapB, noiseMapBTrans, SAMPLE_CNT, SCAN_CNT);//512*1024矩阵专置为1024*512
    for (int i=0; i<SCAN_CNT*SAMPLE_CNT; i++) {
        noiseMap[i] = (noiseMapA[i] > noiseMapBTrans[i]) ? noiseMapA[i] : noiseMapBTrans[i];
        if (bitMapA[i] && bitMapBTrans[i]) {
            bitMap[i] = 1.0;
        }
        else {
            bitMap[i] = 0.0;
        }
    }
    free(bitMapA);
    free(bitMapB);
    free(noiseMapA);
    free(noiseMapB);
    free(dbDataTransponse);
    free(bitMapBTrans);
    free(noiseMapBTrans);
}