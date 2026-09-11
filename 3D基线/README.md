# 三维 OS-CFAR 基线说明

本目录保存三维 OS-CFAR 基线实现。输入数据布局为 `[Z_RANGE][X_RANGE][Y_RANGE]`，程序分别沿 Y、X、Z 三个方向执行一维 OS-CFAR，再合并三个方向的结果，输出三维二值目标位图。

## 文件说明

- `osCfar.c`：一维检测、噪声估计、三维转置、三方向检测和调试辅助函数。
- `osCfar.h`：`CFARConfig` 参数结构体及三维检测主接口。


## 处理流程

1. 对每个固定的 `(z, x)`，沿 Y 方向执行一维 OS-CFAR，得到 `bitMap_zx`。
2. 将每个 Z 平面转置，对每个固定的 `(z, y)`，沿 X 方向检测，得到 `bitMap_zy`。
3. 将数据从 Z-X-Y 布局转换为 X-Y-Z 布局，对每个固定的 `(x, y)`，沿 Z 方向检测，得到 `bitMap_xy`。
4. 将三个方向的结果映射回统一坐标并合并，生成最终三维位图。

当前基线代码使用逻辑“或”合并：任一方向检出即将该单元标记为目标。

## 配置参数

```c
typedef struct {
    int CFAR_AVG_LEFT;
    int CFAR_AVG_RIGHT;
    int CFAR_GUARD_INT;
    int CFAR_OS_KVALUE;
    float CFAR_THR_KSCALE;
    float CFAR_THR_SCALE;
} CFARConfig;
```

- `CFAR_AVG_LEFT`、`CFAR_AVG_RIGHT`：左右参考单元数量。
- `CFAR_GUARD_INT`：单侧保护单元数量。
- `CFAR_OS_KVALUE`：参考窗排序后使用的噪声单元下标，从 0 开始。
- `CFAR_THR_KSCALE`：叠加在 OS 噪声估计值上的检测门限。
- `CFAR_THR_SCALE`：当前主检测路径中未实际参与判决，属于保留参数。

## 主要函数

### `localMaxDetect`

```c
inline void localMaxDetect(const float *data, int len, bool *out);
```

检测一维序列中的局部极大值，左右边界按循环方式处理。

### `osCfarNoiseFinder`

```c
void osCfarNoiseFinder(
    const float *data,
    int col,
    CFARConfig *config,
    float *noiseOut
);
```

收集检测单元两侧的参考窗数据，对窗口进行稳定升序排序，并取 `CFAR_OS_KVALUE` 指定的值作为噪声估计。

### `cfarDetect1d`

```c
void cfarDetect1d(
    const float *src,
    int col,
    CFARConfig *config,
    bool *bitMap
);
```

一维检测核心函数。先循环扩展输入序列，再进行局部极大值和均值预筛选，最后比较目标值与 OS 噪声门限，输出 0/1 检测结果。

### `mat_zxy_transpose_xyz`

```c
void mat_zxy_transpose_xyz(
    const float (*inputData)[X_RANGE][Y_RANGE],
    float (*outData)[Y_RANGE][Z_RANGE]
);
```

将三维数据从 `[Z][X][Y]` 重排为 `[X][Y][Z]`，便于沿 Z 方向连续检测。

### `osCfarDetectMmw3D`

```c
void osCfarDetectMmw3D(
    const float (*inputData)[X_RANGE][Y_RANGE],
    CFARConfig *config_zy,
    CFARConfig *config_zx,
    CFARConfig *config_xy,
    int frame,
    uint16_t (*bitMap)[X_RANGE][Y_RANGE]
);
```

三维检测主入口。

- `inputData`：输入三维能量数据，维度为 `[Z_RANGE][X_RANGE][Y_RANGE]`。
- `config_zy`：沿 X 方向检测时使用的配置。
- `config_zx`：沿 Y 方向检测时使用的配置。
- `config_xy`：沿 Z 方向检测时使用的配置。
- `frame`：帧编号；当前函数主体未使用，仅为上层调用保留。
- `bitMap`：输出三维二值检测结果，由调用方预先分配空间。
- 返回值：无；检测点数量通过日志 `os_cfar:num=...` 输出。


## 使用注意

- 源码依赖 Intel MKL，并使用 AVX2、OpenMP、POSIX 内存对齐及目录外的 `osCfarOpt.h`，编译时需配置相应头文件、编译选项和链接库。
- 调用方需为 `bitMap` 分配 `Z_RANGE * X_RANGE * Y_RANGE` 个 `uint16_t` 元素。

