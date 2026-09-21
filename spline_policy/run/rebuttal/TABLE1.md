# M1：等参数表示误差

```bash
cd spline_policy_release/spline_policy
python run/rebuttal/table1_basis.py --data ../../others/LASAHandwritingDataset/DataSet
python -m pytest run/rebuttal/test_basis_reconstruction.py -q
```

`--data` 指向原始 LASA `.mat` 目录。结果默认保存到 `outputs/m1_table1/`：`table1.pdf/png/svg/tex/md`、逐演示 `per_demo.csv`、汇总 `summary.csv`、完整重构及系数 `reconstructions.npz`、含数据和代码哈希的 `manifest.json`。

## 协议

- 30 类 × 7 条演示，保留每条原始 1000 点及坐标，不归一化、不降采样。每个方法独立拟合每条演示。
- 每维自由系数 K = 3/7/12/17/22；二维总共 2K 个实数。横轴不是完整 FAST 的 BPE token 数。
- 主指标为每条轨迹的 `mean_t ||p_hat[t] - p[t]||_2`，表中统计 210 条的均值 ± 总体标准差（ddof=0），单位为 LASA 原始坐标单位。
- 同时保存 RMSE、最大点误差、`||P_hat-P||_F / T` 和相对原轨迹包围盒对角线的误差，避免将论文公式的矩阵范数写法与逐点平均混用。
- RCFS 基函数及样条使用 float64 最小二乘、上游的 `1e-8` ridge；通过增广矩阵求解，避免形成正规方程。FAST 连续版直接使用官方正交变换。

## 方法与来源

RCFS 基函数定义参考 [MP.py，f4c96d6](https://github.com/idiap/rcfs/blob/f4c96d66cab69f8612a3d380861a909bc6904ad0/python/MP.py)：

- Piecewise：每段 `ceil(T/K)` 个样本，末段截断。
- B.P.：单条 K−1 阶 Bernstein 曲线。
- RBF：`exp(-100 * (t - mu_k)^2)`，中心在 [0,1] 等距分布；保留上游固定宽度。
- Fourier：先拼接原轨迹及其逆序，在 2T 点上拟合 `cos(2*pi*k*t)`，最后保留前 T 点；保留上游镜像处理。

FAST 参考 [官方处理器，ec4d7aa](https://huggingface.co/physical-intelligence/fast/blob/ec4d7aa71691cac0b8bed6942be45684db2110f4/processing_action_tokenizer.py)：沿时间轴调用 `dct(..., norm="ortho")`，保留前 K 个频率、补零后 `idct`。为回应 M1 的连续等维基对照，省略量化和 BPE；固定截断到 K 是本实验的预算设置，并非声称官方 tokenizer 使用固定 K。Fourier 与 DCT 的镜像网格不同，数值接近但不是重复调用同一实现。

主表只保留 **Q.S. (ours)**：复用 release 的 `QuadraticSpline.C/BC`，使用 K−1 段、段间 C1 和终端零速度约束。消去一个系数后仍有每维 K 个自由参数，对应流场的终点停止条件。未加终端约束的旧拟合保留在原始数据的 `spline` 项；主表使用 `spline_rest` 项。

主表将 Fourier 和 FAST 连续版合并为 **Fourier / DCT (FAST-cont.)**，统一报告 DCT 的数值。RCFS Fourier 仍保留在 CSV/NPZ 中供复现核对；主表共五行，原始数据保留七种实现。

## 表格的含义

表格最后一列是最近点求解方式，不是计时结果。Q.S. 每段只需解至多三次方程并检查端点，可直接接入现有解析流场；B.P. 的驻点方程一般为 2K−3 阶，RBF/Fourier/DCT 也能通过数值投影构建流场。Piecewise 是离散常值表示，不具备平滑路径切向。

这张表回答表示误差及解析流场构造的取舍，配合 `outputs/m1_projection/` 的投影实验使用；它不替代等维策略训练成功率实验，也不支持“所有替代基无法构建流场”。参考 [RAL Table I](https://arxiv.org/pdf/2504.09705#page=4)。

原始参考拟合复现 RAL Table I 全部 25 个单元格（两位小数一致）。当前主表 K=12 时 Q.S. / DCT 为 **0.21±0.09 / 0.19±0.08**；相对各演示包围盒对角线的平均误差分别为 **0.402% / 0.350%**。样条误差较小且有解析投影，但不是所有预算下最准确。

13 项数值检查通过；RCFS 原函数独立比较及终端约束残差见 `outputs/m1_table1/verification.json`。生成的 LaTeX 已通过 IEEEtran 编译。
