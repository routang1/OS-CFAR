# OS-CFAR

本仓库保存 OS-CFAR（Ordered-Statistics Constant False Alarm Rate，有序统计恒虚警检测）的二维基线、三维基线和 AVX-512 优化实现。代码以一维 OS-CFAR 为核心，沿二维或三维数据的各个轴进行检测，并合并各方向结果。

## 当前实现

- **1D CFAR**：从待检测单元两侧选取训练单元，排除保护单元，通过排序统计量或等价的比较计数完成门限判决，并结合局部极大值筛选。
- **2D 基线**：沿原矩阵行方向检测；转置矩阵后再按行检测原矩阵列方向；两个方向的检测位图取逻辑与，噪声图取较大值。
- **3D 基线**：输入布局为 `[Z][X][Y]`，分别沿 Y、X、Z 三个方向检测，三个方向的位图取逻辑或。
- **3D 优化版**：使用 AVX-512 加速参考窗比较、门限判断和结果写回；X 方向采用跨行直接访问，取消基线中的逐 Z 平面二维转置；Z 方向仍通过 `[Z][X][Y] -> [X][Y][Z]` 轴置换获得连续数据。

默认三维尺寸由 `osCfarOpt.c` 定义：`Z_RANGE=31`、`X_RANGE=61`、`Y_RANGE=51`。这些宏可以在编译时通过 `-D` 覆盖，但输入数据和配置长度必须与新尺寸一致。

## 目录结构

```text
OS-CFAR/
├── 2D基线/
│   ├── osCfar.c              # 1D/2D 标量检测、二维转置和结果合并
│   ├── osCfar.h              # 2D CFARConfig 与接口声明
│   └── README.md             # 2D 基线说明
├── 3D基线/
│   ├── osCfar.c              # 三方向基线检测、AVX2 分块转置和轴置换
│   ├── osCfar.h              # 3D CFARConfig 与接口声明
│   └── README.md             # 3D 基线说明
├── .vscode/
│   └── c_cpp_properties.json # 当前 GCC IntelliSense 配置
├── osCfarOpt.c               # 可执行的 3D AVX-512 优化版及 main 入口
├── OSCFAR使用手册.md
└── OSCFAR点云检测加速技术文档.md
```

仓库当前没有测试数据、自动化测试、Makefile 或 CMake 工程。

## 转置与数据布局

二维基线中的 `matriTransponse` 和 `boolMatriTransponse` 将行列互换，使原来跨行、不连续的列方向数据可以作为连续行处理。

三维基线中的 `mat_transpose_avx2` 使用 `transpose_8x8_avx2` 对每个 Z 平面进行 8×8 分块转置，布局从 `[Z][X][Y]` 变为 `[Z][Y][X]`，用于沿 X 方向检测；`mat_zxy_transpose_xyz` 将数据变为 `[X][Y][Z]`，用于沿 Z 方向检测。

优化版保留 Z 方向的 `[Z][X][Y] -> [X][Y][Z]` 轴置换。X 方向由 `vertical_cfar_direct_avx512` 直接跨行读取同一列的参考单元，因此不再生成 `[Z][Y][X]` 中间矩阵。

## 依赖与编译

### 优化版

`osCfarOpt.c` 仅依赖 C 标准库、`<sys/time.h>` 和 x86 intrinsic 头文件，不调用 oneMKL 或 OpenMP。编译器需要支持 AVX2、AVX-512F 和 AVX-512BW；运行机器的 CPU 与操作系统也必须支持这些指令集。

在仓库根目录使用 GCC 编译：

```bash
gcc -O3 -std=c17 -mavx2 -mavx512f -mavx512bw -Wall -Wextra osCfarOpt.c -o osCfarOpt.exe
```

该命令已使用仓库当前源码和 MSYS2 UCRT64 GCC 完成编译验证。当前编译会报告若干未使用变量、未使用函数和未使用参数警告，但可以成功生成程序。

### 2D/3D 基线

2D 基线引用了当前仓库未包含的 `commDef.h`，其中应提供 `SAMPLE_CNT`、`SCAN_CNT` 等尺寸定义。3D 基线同样引用 `commDef.h`，还引用当前仓库未包含的 `osCfarOpt.h`；它实际调用 oneMKL 的 `mkl_malloc`、`mkl_somatcopy` 等接口，并包含 OpenMP、pthread 和 AVX2/AVX-512 相关头文件。

因此，当前仓库不能单独编译这两个基线目录。需要先从原集成工程补齐公共头文件和尺寸定义，并配置 oneMKL、OpenMP及 pthread 后，才能确定完整、可复现的基线编译命令。

## 运行优化版

`main` 固定处理 24 帧文本输入：

```text
data_trace6_log/r_log1.txt
...
data_trace6_log/r_log24.txt
```

每帧需要提供 `Z_RANGE * X_RANGE * Y_RANGE` 个可由 `%f` 读取的浮点数。读取循环顺序为 Z、Y、X，数据写入 `inputData[z][x][y]`。程序将 `uint16_t[Z_RANGE][X_RANGE][Y_RANGE]` 检测位图写入：

```text
c_outputs/r_log01_det.bin
...
c_outputs/r_log24_det.bin
```

运行前需要准备输入目录和文件，并创建输出目录；程序本身不会创建 `c_outputs`：

```bash
mkdir -p c_outputs
./osCfarOpt.exe
```

测试数据目前不在仓库中，`.gitignore` 也没有全局排除 `.txt` 或 `.bin`，以免将后续需要纳入版本控制的数据误排除。程序生成的 `c_outputs/` 被忽略。

## 文档

- [OSCFAR使用手册.md](OSCFAR使用手册.md)：优化版接口、参数、数据布局和调用注意事项。
- [OSCFAR点云检测加速技术文档.md](OSCFAR点云检测加速技术文档.md)：从基线到 AVX2、AVX-512、Padding 和直接列访问的优化过程。
- [2D基线/README.md](2D基线/README.md)：二维检测流程与接口说明。
- [3D基线/README.md](3D基线/README.md)：三维基线流程、接口和依赖说明。
