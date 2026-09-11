#ifndef OS_CFAR_H
#define OS_CFAR_H

#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int CFAR_AVG_LEFT;     // 左侧参考窗点数（毫米波雷达通常取2~4）
    int CFAR_AVG_RIGHT;    // 右侧参考窗点数
    int CFAR_GUARD_INT;    // 单侧保护窗点数（避免目标能量污染参考窗）
    int CFAR_OS_KVALUE;    // OS-CFAR 的 K 值（0索引，通常取参考窗长度的1/2）
    double CFAR_DET_THR;   // 检测阈值（毫米波雷达通常取5~10 dB）
} CFARConfig;

void selectionSort(double *arr, int len);
void osCfarDetectMmw2D(double *dbData, CFARConfig* config, double *bitMap, double *noiseMap);
void localMaxDetect(const double *data, int len, bool *out);
void osCfarNoiseFinder(const double *data, int col, CFARConfig* config, double *noiseOut);
void osCfarDetectMmw1D(const double *dbData, int row, int col, CFARConfig* config, bool *bitMap, double *noiseMap);
#ifdef __cplusplus
}
#endif
#endif