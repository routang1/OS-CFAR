# 二维 OS-CFAR 基线说明

本目录保存二维 OS-CFAR（有序统计恒虚警检测）的基线实现。程序分别沿二维矩阵的行方向和列方向进行一维检测，仅将两个方向都检出的单元标记为目标。

## 文件说明

- `osCfar.c`：排序、噪声估计、一维检测、矩阵转置及二维检测实现。
- `osCfar.h`：`CFARConfig` 参数结构体和主要函数声明。
- `commDef.h`：目录外公共头文件，需提供 `SAMPLE_CNT`、`SCAN_CNT` 等数据尺寸定义。

## 处理流程

1. 沿原始矩阵的每一行执行一维 OS-CFAR。
2. 将矩阵转置，再沿转置矩阵的每一行执行检测，相当于检测原矩阵的列方向。
3. 将列方向结果转回原始布局。
4. 对两个方向的检测结果取逻辑“与”，得到最终目标位图；两个方向的噪声值取较大值作为最终噪声图。

## 配置参数

```c
typedef struct {
    int CFAR_AVG_LEFT;
    int CFAR_AVG_RIGHT;
    int CFAR_GUARD_INT;
    int CFAR_OS_KVALUE;
    double CFAR_DET_THR;
} CFARConfig;
```

- `CFAR_AVG_LEFT`、`CFAR_AVG_RIGHT`：检测单元左右两侧的参考单元数量。
- `CFAR_GUARD_INT`：单侧保护单元数量。
- `CFAR_OS_KVALUE`：参考窗排序后选取的噪声单元下标，从 0 开始。
- `CFAR_DET_THR`：在噪声估计值上叠加的检测门限，输入为 dB 数据时单位为 dB。

## 主要函数

### `selectionSort`

```c
void selectionSort(double *arr, int len);
```

对参考窗数据进行升序选择排序，供 OS-CFAR 选取第 `K` 个有序统计量。

### `localMaxDetect`

```c
void localMaxDetect(const double *data, int len, bool *out);
```

比较每个单元与左右相邻单元，仅标记局部极大值。数组边界按循环方式处理。

### `osCfarNoiseFinder`

```c
void osCfarNoiseFinder(
    const double *data,
    int col,
    CFARConfig *config,
    double *noiseOut
);
```

为一行数据中的每个检测单元建立左右参考窗，排序后取 `CFAR_OS_KVALUE` 指定的值作为背景噪声估计。参考窗在边界处循环取值。

### `osCfarDetectMmw1D`

```c
void osCfarDetectMmw1D(
    const double *dbData,
    int row,
    int col,
    CFARConfig *config,
    bool *bitMap,
    double *noiseMap
);
```

逐行执行一维 OS-CFAR。只有同时满足“当前值不低于噪声加门限”和“当前点为局部极大值”的单元才会被标记。

### `matriTransponse` / `boolMatriTransponse`

```c
void matriTransponse(double *in, double *out, int row, int col);
void boolMatriTransponse(bool *in, bool *out);
```

分别转置浮点矩阵和布尔检测矩阵，用于切换行、列检测方向。

### `osCfarDetectMmw2D`

```c
void osCfarDetectMmw2D(
    double *dbData,
    CFARConfig *config,
    double *bitMap,
    double *noiseMap
);
```

二维检测主入口。输入、输出均按 `SAMPLE_CNT × SCAN_CNT` 的连续一维内存保存。`bitMap` 输出 0 或 1，`noiseMap` 输出两个检测方向中较大的噪声估计值。

## 使用注意

- 调用方需为 `bitMap` 和 `noiseMap` 分配 `SAMPLE_CNT * SCAN_CNT` 个元素的空间。
- 必须保证 `0 <= CFAR_OS_KVALUE < CFAR_AVG_LEFT + CFAR_AVG_RIGHT`。
- 行、列方向共用同一套 `CFARConfig`；若需要不同参数，应修改主接口或分别调用一维检测。


