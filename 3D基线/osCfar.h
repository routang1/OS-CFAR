#ifndef OS_CFAR_H
#define OS_CFAR_H

#include "commDef.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

typedef struct {
    int CFAR_AVG_LEFT;     // 左侧参考窗点数（毫米波雷达通常取2~4）
    int CFAR_AVG_RIGHT;    // 右侧参考窗点数
    int CFAR_GUARD_INT;    // 单侧保护窗点数（避免目标能量污染参考窗）
    int CFAR_OS_KVALUE;    // OS-CFAR 的 K 值（0索引，通常取参考窗长度的1/2）
    float CFAR_THR_KSCALE;
    float CFAR_THR_SCALE;
} CFARConfig;

void osCfarDetectMmw3D(const float (*inputData)[X_RANGE][Y_RANGE], CFARConfig* config_zy, CFARConfig* config_zx, CFARConfig* config_xy, int frame, uint16_t (*bitMap)[X_RANGE][Y_RANGE]);
#endif