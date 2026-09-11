# osCfarOpt.c 交接文档

## 1. 交接文件

- `osCfarOpt.c`


## 2. 数据布局

核心函数接收的输入数组布局是：

```text
float inputData[Z_RANGE][X_RANGE][Y_RANGE]
```

当前尺寸是：

```c
X_RANGE = 61
Y_RANGE = 51
Z_RANGE = 31
```

也就是说，访问单点时使用：

```c
inputData[z][x][y]
```

输出 bitmap 的布局是：

```text
uint16_t bitMap[Z_RANGE][X_RANGE][Y_RANGE]
```

输出含义：

```c
bitMap[z][x][y] == 1  // 该点被检出
bitMap[z][x][y] == 0  // 该点未检出
```

输入值应当已经是当前算法需要的 log 域数据；当前 `osCfarOpt.c` 内部没有再做 `log2f()` 转换。如果上游原始数据是 `double`，交给本函数前需要显式转为 `float`，或者同步把接口和内部计算类型改成 `double`。





## 3. 算法状态

当前算法使用三方向 CFAR，三个配置按代码实际使用方向如下：

```c
config_zy: scan X, len=X_RANGE, left=20, right=20, guard=4, K=19, kscale=0.53, scale=-1.53
config_zx: scan Y, len=Y_RANGE, left=16, right=16, guard=4, K=15, kscale=0.53, scale=-1.53
config_xy: scan Z, len=Z_RANGE, left=8,  right=8,  guard=4, K=7,  kscale=0.53, scale=-1.53
```

字段含义：

- `len`：当前检测方向的长度，X 方向为 `X_RANGE`，Y 方向为 `Y_RANGE`，Z 方向为 `Z_RANGE`。
- `CFAR_AVG_LEFT` / `CFAR_AVG_RIGHT`：CUT 左右两侧参考窗点数，不包含保护窗和 CUT 本身。
- `CFAR_GUARD_INT`：单侧保护窗点数，代码中的保护区总长度为 `2 * CFAR_GUARD_INT + 1`，其中包含 CUT。
- `CFAR_OS_KVALUE`：OS-CFAR 的排序/计数门限 K。当前实现通过统计参考窗中 `ref <= cur - CFAR_THR_KSCALE` 的数量，并判断 `count > K`。
- `CFAR_THR_KSCALE`：OS-CFAR 幅度偏置，当前比较门限为 `cur - CFAR_THR_KSCALE`。
- `CFAR_THR_SCALE`：原最大值门限使用的偏置，形式为 `peak_limit = row_or_col_max + CFAR_THR_SCALE`。当前行最大值门限已经删除，实际 gate 只用均值门限，因此该字段目前保留但不参与最终检测。
- `CFAR_AVG_OFFSET`：均值门限偏置，当前为 `3.0f`，实际 gate 为 `cur >= mean + CFAR_AVG_OFFSET`。

当前保留局部最大值检测：

```c
cur >= 前一个点 && cur >= 后一个点
```

三方向检测结果最后做 OR：

```c
bitMap = bitMap_zx | bitMap_zy | bitMap_xy
```



## 4. 主要函数调用方式

 3D 检测入口：

```c
void osCfarDetectMmw3D(const float (*inputData)[X_RANGE][Y_RANGE],
                       CFARConfig* config_zy,
                       CFARConfig* config_zx,
                       CFARConfig* config_xy,
                       int frame,
                       uint16_t (*bitMap)[X_RANGE][Y_RANGE]);
```

参数说明：

- `inputData`：输入三维数据，实际内存布局必须是 `float[Z_RANGE][X_RANGE][Y_RANGE]`。
- `config_zy`：X 方向 CFAR 配置，`.len` 必须等于 `X_RANGE`。
- `config_zx`：Y 方向 CFAR 配置，`.len` 必须等于 `Y_RANGE`。
- `config_xy`：Z 方向 CFAR 配置，`.len` 必须等于 `Z_RANGE`。
- `frame`：当前没有参与检测逻辑，只是保留接口参数；可以传帧号，也可以传 `0`。
- `bitMap`：输出三维 bitmap，实际内存布局必须是 `uint16_t[Z_RANGE][X_RANGE][Y_RANGE]`。

配置示例：

```c
CFARConfig config_zy = {
    .len = X_RANGE,
    .CFAR_AVG_LEFT = 20,
    .CFAR_AVG_RIGHT = 20,
    .CFAR_GUARD_INT = 4,
    .CFAR_OS_KVALUE = 19,
    .CFAR_THR_KSCALE = 0.53,
    .CFAR_THR_SCALE = -1.53
};

CFARConfig config_zx = {
    .len = Y_RANGE,
    .CFAR_AVG_LEFT = 16,
    .CFAR_AVG_RIGHT = 16,
    .CFAR_GUARD_INT = 4,
    .CFAR_OS_KVALUE = 15,
    .CFAR_THR_KSCALE = 0.53,
    .CFAR_THR_SCALE = -1.53
};

CFARConfig config_xy = {
    .len = Z_RANGE,
    .CFAR_AVG_LEFT = 8,
    .CFAR_AVG_RIGHT = 8,
    .CFAR_GUARD_INT = 4,
    .CFAR_OS_KVALUE = 7,
    .CFAR_THR_KSCALE = 0.53,
    .CFAR_THR_SCALE = -1.53
};
```

最小调用示例：

```c
float (*inputData)[X_RANGE][Y_RANGE] =
    (float (*)[X_RANGE][Y_RANGE])_mm_malloc(
        (size_t)Z_RANGE * X_RANGE * Y_RANGE * sizeof(float), 64);

uint16_t (*bitMap)[X_RANGE][Y_RANGE] =
    (uint16_t (*)[X_RANGE][Y_RANGE])_mm_malloc(
        (size_t)Z_RANGE * X_RANGE * Y_RANGE * sizeof(uint16_t), 64);

if (inputData == NULL || bitMap == NULL) {
    /* handle allocation failure */
}

/* 上层工程在这里填充 inputData[z][x][y] */

osCfarDetectMmw3D(inputData,
                  &config_zy,
                  &config_zx,
                  &config_xy,
                  frame_id,
                  bitMap);

/* 上层工程在这里读取 bitMap[z][x][y] */

_mm_free(inputData);
_mm_free(bitMap);
```

如果只想对单个 `z` 高度下的二维 `X x Y` 矩阵做检测，可以调用二维入口：

```c
void zhwx_oscfar(const float (*input)[Y_RANGE],
                 bool (*bitMap_zx)[Y_RANGE],
                 bool (*bitMap_zy)[X_RANGE],
                 const CFARConfig* config_zx,
                 const CFARConfig* config_zy);
```

完整 3D 检测一般不需要上层直接调用 `zhwx_oscfar()`，因为 `osCfarDetectMmw3D()` 内部已经会对每个 `z` 切片调用它，再补充 Z 方向检测，最后合成总 bitmap。

## 5. 测试函数说明

以下函数是当前文件里的本地测试辅助，不是正式算法接口：

- `main()`：批量读取测试 txt、计时、调用检测、写出 bin。
- `read_float_3D_data()`：按当前测试 txt 格式读取数据。
- `save_bin()`：把测试输出写成 bin。
- `timeval_diff_ms()`：测试计时。



## 6. 编译说明

当前实现使用 AVX512 intrinsic 和 `_mm_malloc/_mm_free`，编译参数至少需要包含：

```text
-O3 -mavx2 -mavx512f -mavx512bw -std=c99
```


