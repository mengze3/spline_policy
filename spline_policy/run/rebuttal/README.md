# M1：最近点估计与执行轨迹

作图统一遵循 [论文作图准则](../../../FIGURE_STYLE.md)，字体、配色与布局以论文和 PPT 为依据。

本实验现在只保留三个 Python 文件；所有旧版图、搜索日志、GN 实验和重复绘图脚本已删除。

| 文件 | 用途 |
| --- | --- |
| `fig6_projection.py` | 同一条二次 S 样条、两种最近点估计、执行、分辨率／步长对照、独立求根核验 |
| `search_moving_approach.py` | 扫描初始点与参数，分别记录连续反向次数、期间净位移、是否恢复到达目标 |
| `plot_projection_comparison.py` | 一张长条图：左半 Numerical，右半 Analytical，各含完整空间轨迹和原始位置时间曲线 |

独立的等参数表示误差实验保留在 [TABLE1.md](TABLE1.md)，代码为 `table1_basis.py`、`basis_reconstruction.py`、`test_basis_reconstruction.py`，结果仍在 `outputs/m1_table1/`。

## 复现

在 `spline_policy_release/spline_policy` 下运行：

```bash
python run/rebuttal/fig6_projection.py
python run/rebuttal/plot_projection_comparison.py
```

当前只使用 `outputs/m1_projection/`：

- `comparison.pdf/png/svg`：同尺度的左右对比；所有背景均为解析流场，沿用论文配色。
- `rollouts.npz`、`results.json`：六组完整执行、参数、统计与解析最近点核验。
- `projection_queries.npz`：两种投影使用同样的真实执行状态及对应最近点。
- `caption.txt`：独立图注；时间曲线直接标注到达时刻。
- `search.json`：下面命令的搜索设置、统计及候选，不保存每次尝试的图。

```bash
python run/rebuttal/search_moving_approach.py
```

## 当前结果及限制

保留原 S 曲线：读取 `S.zarr` 第一条演示，除以 512，拟合六段二次样条并约束终端切向为零。
Numerical 在整条曲线上取 **20 个等参数采样点（含端点）**，每步选欧氏距离最近的一个；Analytical 在所有曲段的有效三次方程根和端点中选连续最近点。

两者均从 **(0.75, 1.05)** 出发，吸附系数 **0.1**、步长 **0.08 s**、速度 **0.5**、时间尺度 **6 s**，执行 **16 s**。到终点距离 ≤0.005 后保持位置。
这是重新选择的参数组合；**吸附系数低于上一轮的 0.4**，不能描述成增强吸附的效果。

控制律为 `F = (λ(q-p) + f′(phase)/T) / (1 + λ||q-p||)`，归一化为速度 0.5 后进行固定步长 Euler 积分。除最近点估计外，两个对照设置完全相同。

| 设置 | 数值到达时间 | 解析到达时间 |
| --- | --- | --- |
| N=20，dt=0.08 | 8.24 s | 3.44 s |
| N=40，dt=0.08 | 3.52 s | 3.44 s |
| N=20，dt=0.04 | 3.64 s | 3.64 s |

主例在到达曲线之前共反向 61 次。最长连续段为第 33–91 步的 **59 次反向**；包围该段的第 32–92 步覆盖 **2.56–7.36 s**，整段之前的执行折线距离曲线下界为 **0.13875**。
这一窗口净位移只有 **0.00413**，主要是在局部往返，之后恢复前进并到达目标。主图显示完整空间轨迹，并直接画原始 x(t)，展示 8.24/3.44≈**2.40 倍**的到达耗时；没有添加波形或变形坐标。

**这仍不是用户希望的大幅持续摆动着前进的案例。** 默认搜索找到一个持续反向且最终恢复的案例，没有找到同时通过窗口净位移 ≥两个步长的案例。当前搜索的有限范围不能推出不存在其他情况。
提高采样密度或减小积分步长会消除本例的长时间往返；本例不能证明所有数值方法、DCT 或 FAST 必然失效。解析轨迹自身的一次明显转向也完整保留。

主图两条轨迹共 402 个状态使用独立数值多项式求根核验连续距离，最大平方距离差为 **6.94e-17**。图中最近点箭头取数值执行的第 32、33 步，两侧使用相同查询点。

## M3.3：Table III 六任务拟合误差（2026-09-26）

`fitting_residual.py` 为单个 CPU 脚本，直接使用原二次 spline 解码基（6段、8参数/维、16点、自由端点），核验 release/3D 两版基相同。对所有完整 stride-1 窗口拟合，不补边、不跨 episode；按任务各动作坐标全数据范围归一化。每条示范汇总窗口平方误差后开根，再报告示范间 mean±sampleSD(ddof=1)，另报所有窗口P95。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run/rebuttal/fitting_residual.py --fetch-robomimic
```

`--fetch-robomimic` 仅在缓存缺失时，按HTTP Range读取官方 Diffusion Policy 的 Can/PH、Transport/PH lowdim absolute-action ZIP成员，在内存解压并校验CRC；只保留 `data/m3_robomimic_actions.npz`（9.4MiB）。Can使用官方同PH设置的lowdim动作数据，未核验原image训练文件哈希。其余四任务读现有zarr。数据原件不改。

结果仅 `outputs/m3_fitting/windows.csv` 和 `results.json`：138527窗口/746示范，含每窗RMSE/二阶差分、每示范统计、来源/哈希/归一化范围及数值核验。未加载策略权重、不使用GPU；这是离线最佳拟合诊断，不是任务成功率或TableIII分差的因果解释。

2026-09-26 M3.3 图：同一脚本加 `--plot-examples`，输出 `fitting_examples.pdf/png/svg`。左侧保持六任务 mean±sampleSD/P95。右侧改为 Adroit Door：全1700窗口中最大28维RMSE为12.46479%，episode/start=0/5；显示该窗口误差最大的第23维（index22，单维RMSE22.00512%）。好例为同维变化幅度≥全数据范围50%的510个窗口中整体误差最低者，episode/start=12/42，整体/单维RMSE为0.38275%/0.53170%。两例共享时间轴和归一化动作轴，不再作为二维位置或像素展示。选择规则与28维数组存入原results.json；左侧统计与原始数据不变。

## SafeLIBERO Level I：抓取放置与安全走廊

入口：[safelibero_corridor.py](safelibero_corridor.py)。只新增这一份实验代码，复用 release 的 `QuadraticSpline`。当前有限三维走廊演示在 `outputs/safelibero_finite3d_20260928/`；旧半空间演示 `outputs/safelibero_level1_demo_20260928/` 的全部 69 个文件原样保留。

使用公开 SafeLIBERO-Object「橙汁入篮」Level I 第 **0** 个初始状态，以及原 LIBERO 的 **demo_0**。障碍为靠近橙汁的架高酒瓶，场景与障碍位置均未人工改动。它是**公开示范的轨迹适配演示**，不是新训练的 Spline Policy，也不是成功率实验。

### 当前版本：有限三维走廊，重新执行的配对结果

`outputs/safelibero_finite3d_20260928/corridor_definition.npz` 在优化前保存 14 个相互重叠的有限三维区域。区域来自原始名义曲线和障碍几何，不依赖优化结果。每个区域的 x/y/z 上下界全部施加到对应样条段的 Bézier 控制点上；另外逐面收紧 25 mm 作为跟踪余量，固定起终点所在面的余量按端点距离缩小。相邻收紧区域仍重叠。QP 失败即报错，不回退到无约束曲线。

绘图直接读取这些边界，`display_geometry.npz` 与求解前定义逐元素一致，不再根据求解结果补画外墙。12 个独立不可行测试逐一把外边界／内边界的六个方向移至排除固定起点，均正确报告不可行。连续曲线的安全性用控制凸包和独立二次曲线极值核验，C0/C1 残差为 0 / 7.46e-14。

| 配对执行 | 完成入篮 | 接触采样帧 | 记录的末端走廊越界点 | 最近实际关节限位余量 |
|---|---|---|---|---|
| 无走廊约束 | 否 | 16 | 36 / 408 | 0.611 rad |
| 有限三维走廊 | 是 | 0 | 0 / 408 | 0.613 rad |

两侧从相同官方初始状态重新执行，共享控制器、姿态／夹爪指令和位置起终点。有走廊侧最大跟踪误差为 66.6 mm，但全部 408 个记录点仍在其对应的外层有限区域内；逐面预留 25 mm 不是跟踪误差的保证上界。实际执行核验是 20 Hz 采样，不代表每个物理子步或整个机械臂的连续安全证明。酒瓶最大 L1 位移为 0.00139 mm。`audit.json`、`preflight.json`、求解前定义与源码快照保留完整证据；未拼接旧数据。

```bash
# 新输出目录必须不存在，防止覆盖已有实验。
MUJOCO_GL=egl NUMBA_CACHE_DIR=/tmp/safelibero-numba OMP_NUM_THREADS=1 \
  ../../.venvs/safelibero/bin/python run/rebuttal/safelibero_corridor.py \
  --output outputs/safelibero_finite3d_new
# 绘图读取同一次新实验的真实边界与物理录像。
../../.venvs/safelibero/bin/python run/rebuttal/safelibero_corridor.py \
  --compare outputs/safelibero_finite3d_20260928 \
  --output outputs/safelibero_finite3d_20260928/corridor_comparison
```

最终图片为 `corridor_comparison/comparison_407.png`，视频为该目录 `comparison.mp4`。沿用论文／PPT 的轨迹配色、大字、紧凑分区和白色发光光圈；圈注酒瓶使用每帧记录的真实位置和四元数。两侧采用相同相机、裁剪与比例，三维视图中的棕色障碍为初始包络。

### 历史版本：半空间约束与显示裁剪（原结果保留）

以下旧结果与命令针对半空间版；如需重新执行该旧协议，应使用旧目录的 `source_snapshot.py`，而非已更新的 live 入口。旧图的有限外墙是显示裁剪，不能作为当前有限三维约束的证据。

### 运行

环境使用 `robosuite==1.4.1`、`mujoco==3.2.3`、`bddl==1.0.1`、`easydict==1.9`、`PyOpenGL==3.1.7`，另需 NumPy/SciPy/Torch/CVXPY（CLARABEL）、h5py、PyYAML、Pillow、Matplotlib、imageio/ffmpeg。图中文字使用系统 Arimo 字体。本机依赖在隔离的 `../../.venvs/safelibero` 中，不修改既有训练环境。

```bash
# 从 release/spline_policy 运行；首次 --fetch 下载固定版本场景和单任务示范。
MUJOCO_GL=egl NUMBA_CACHE_DIR=/tmp/safelibero-numba OMP_NUM_THREADS=1 \
  ../../.venvs/safelibero/bin/python run/rebuttal/safelibero_corridor.py \
  --fetch --output outputs/safelibero_level1_demo_20260928
```

资源放在 release 的忽略目录 `data_local/safelibero`，保留上游许可证。上游 commit 为 `2457feed5968ae803926e178c8ce8243b9ecdcf9`；示范 HDF5 的 SHA256 为 `53e40490c62e94c678233eefbfd874a61c93d92c8088f59d426ae6c3fd90d288`。`--preview` 只预览官方场景，`--episode` / `--demo` 明确选择初始状态与示范。

### 方法与核验范围

- 根据当前起点、目标物体和篮子位置，对示范做分阶段平移适配；位置拟合为 48 段 C1 二次样条，姿态用 SLERP。两侧共享适配示范、控制器和夹爪动作。为稳定展示，机械臂和夹爪闭合增量同步重定时 3 倍。
- 从活动酒瓶的碰撞网格计算世界坐标 AABB，向外扩展 80 mm，按每段原参考选择外侧半空间组成任务空间走廊。QP 最小化整条轨迹的改变量，同时约束每段全部三个 Bézier 控制点，保留起终点；该约束覆盖**连续参考曲线**。没有额外 workspace box constraint，也没有手写绕行轨迹。
- 夹爪的观测姿态属于末端连杆，OSC 接受夹爪 site 姿态；入口从当前模型计算固定旋转偏置。两侧都用原 OSC 的绝对目标接口，增益不改；执行靠物理仿真，不把物体焊到夹爪，也不逐帧回写机器人状态。
- Joint limits 施加在独立有界 IK 的关节变量上，距离模型上下限各留 0.02 rad，并检查每个参考目标的位置／姿态误差。IK 使用独立 `MjData`，不改变执行状态。它检查离散目标可达性；不声称已证明连续关节轨迹或全机械臂无碰撞。执行期间另记录实际关节范围。
- 实际任务完成使用官方 `check_success`。保留官方障碍 L1 位移 >1 mm 判据，并额外保存 20 Hz 的 robot/gripper/held-object 对新增障碍的接触记录。采样接触数不是连续碰撞证明；80 mm 是参考的几何余量，不是全机械臂的包络证明。控制误差可能使实际 TCP 离开收紧后的参考走廊，原始误差照实保存。

演示不筛选不同 seed 的最好结果。开发过程的姿态坐标转换、夹爪重定时和余量调试记录另行归档；这次的结论限于同一官方场景与同一示范。它展示新增约束对已有轨迹的适配能力，不能代替训练策略的统计评估。

第 0 场景结果：无约束侧未完成任务，有 16 个采样帧记录到障碍接触；约束侧完成入篮，接触记录为 0，酒瓶最大 L1 位移约 0.00139 mm。约束侧实际关节距最近限位至少 0.613 rad，有界 IK 解至少 0.598 rad。独立核验确认 C0/C1 残差分别为 0 / 8.9e-16，参考控制点满足走廊，982 个上游文件哈希一致。

实际 TCP 相对收紧的参考走廊最大偏离为 **18.7 mm**，虽然该例没有触发碰撞判据，也不能把参考曲线证书写成真实执行的硬约束保证。数值及核验范围见 `audit.json`；初始接口调试失败与后续修正记录保留于 `development_records.zip`。

走廊叠加可直接从已有视频和 NPZ 生成，不重跑物理仿真：

```bash
../../.venvs/safelibero/bin/python run/rebuttal/safelibero_corridor.py \
  --visualize outputs/safelibero_level1_demo_20260928 \
  --output outputs/safelibero_level1_demo_20260928/corridor_3d
```

`corridor_3d/corridor.mp4` 左侧在场景中叠加有侧壁、顶面和底面的半透明三维走廊，右侧用立体视角显示走廊、酒瓶包络与轨迹。相邻显示区域重叠，每个区域覆盖对应样条段的 Bézier 控制凸包，并包含于保存的安全半空间内；显示外边界没有施加到原实验。合并区域的外表面去掉内部重叠面，保留细线表示分段结构。青色／白色是整条参考，紫色是实际执行轨迹，未投影或修正实际轨迹。`display_geometry.npz` 保存显示区域及段索引，`provenance.json` 记录输入哈希。原平面／俯视版保留在 `corridor_view/`，原视频、轨迹和审计不改。新录制保存 `camera.npz`；旧演示使用固定版本官方场景的 agentview 参数。

有／无走廊同步对比（保留已确认的 `corridor_3d/`，不重跑实验）：

```bash
../../.venvs/safelibero/bin/python run/rebuttal/safelibero_corridor.py \
  --compare outputs/safelibero_level1_demo_20260928 \
  --output outputs/safelibero_level1_demo_20260928/corridor_comparison
```

对比视频按 `FIGURE_STYLE.md`、论文 Fig. 8/9 与 PPT 第 9/22/27 页排版：白底、Arimo Bold 大字、灰色分区线，分为 A 任务执行和 B 三维轨迹。去掉地面网格、图内小字脚注和接触计数，仅保留关键结果标签与三项图例。两侧从原录像与 NPZ 重新绘制相同视角、比例和裁剪范围的显示层；已确认的 `corridor_3d/` 独立版本不改。

参考轨迹沿用 PPT 红—紫—蓝相位渐变（`#F1454F → #E388B0 → #C0B4FF → #7296FF`），执行轨迹统一为 `#AD6AEA`。参考／执行线宽相对首个渐变版本均加倍。两侧场景与三维面板使用一致的紧凑裁剪，不截断参考或执行轨迹；原场景统一用 gamma=0.90 轻微提亮，再叠加轨迹和标注。

图内标注参考 PPT 第 7 页：白圈配红／绿柔光，黑字白描边。酒瓶圈表示累计记录到的碰撞状态，橙汁圈独立表示任务结果，最终同时显示 Collision／Failure 与 No collision／Success。橙汁位置来自原始 NPZ；无碰撞酒瓶使用初始位置，倒下酒瓶使用录像手工标定的图像关键帧插值，仅用于圈注，不作为测量数据。标注坐标存入 `provenance.json`。

两侧逐帧同步，各 408 帧、20 Hz，姿态和夹爪指令数组完全相同，位置参考共享起终点。三维示意保留障碍的初始包络，真实酒瓶运动看上排录像。无约束侧 16 个接触采样帧且任务未完成，有走廊侧 0 个接触采样帧且完成入篮；这里只展示同一示范与同一场景的对照。显示裁剪产生的走廊外边界并非新增实验约束。

## DEDO：柔性布料接触与流场执行

`dedo_flow.py` 是独立入口，使用 [DEDO](https://github.com/contactrika/dedo)
（Antonova et al., NeurIPS Datasets and Benchmarks 2021）的 `HangGarment-v1`、
公开 `cloth/apron_0.obj` 网格及 `preset_info.py` 轨迹。上游固定在
`4af113d9342d7d6ec1c5eb5209af53626caa92c0`，不需要训练或 checkpoint。

将两个夹持点的三维位置合为六维状态，拟合 12 段二次样条；每次读取实际位置，
调用 release 原有 `dynamical_system_single_step` 得到速度方向，再由 DEDO 原速度
控制器施力。轨迹的最近点与恢复方向取决于当前位置，不按时间索引播放轨迹。
布料沿用原质量、弹簧、阻尼、摩擦和软体接触；两个夹持点是基准提供的动力学
球形锚点，并非完整机械臂。没有回写执行状态或强制布料贴合参考。

依赖安装到忽略目录，现有训练环境和共享模块不改。以下从 release 根目录准备：

```bash
git clone https://github.com/contactrika/dedo data_local/dedo
git -C data_local/dedo checkout 4af113d9342d7d6ec1c5eb5209af53626caa92c0
../.venvs/safelibero/bin/python -m pip install --target data_local/dedo_deps --no-deps pybullet==3.2.7
```

Python 3.8 环境另需 NumPy、SciPy、Torch、Gym、Matplotlib、Pillow、imageio 和
imageio-ffmpeg；预设回放接口还导入上游的 OpenCV / wandb。图形使用 TinyRenderer，
流场在 CPU 上计算。本机复用 `../.venvs/safelibero`，只额外安装隔离的 PyBullet。
上游原来固定 PyBullet 3.1.7；该版本也已实际测试，但未导出软体接触点。
这里明确使用 3.2.7 以记录布料对挂架／立柱的接触，不宣称逐版本复现基准结果。

```bash
# 从 release/spline_policy 运行，每次指定新输出目录。
../../.venvs/safelibero/bin/python run/rebuttal/dedo_flow.py --output outputs/dedo_flow/nominal
../../.venvs/safelibero/bin/python run/rebuttal/dedo_flow.py --perturb --output outputs/dedo_flow/push
../../.venvs/safelibero/bin/python run/rebuttal/dedo_flow.py \
  --compare outputs/dedo_flow/nominal outputs/dedo_flow/push --output outputs/dedo_flow/figures
# 可选：直接调用官方 build_traj / merge_traj 回放公开预设。
../../.venvs/safelibero/bin/python run/rebuttal/dedo_flow.py \
  --mode preset --steps 200 --output outputs/dedo_flow/preset
```

当前结果在 `outputs/dedo_flow_20260928/`：`nominal` 与 `push` 共享参考、初态和
控制器；后者在约 3 s 时对两个夹持点各施加 +x 方向 16 N 外力，持续 10 个控制
周期（实际区间 3.008–3.168 s）。控制为 62.5 Hz，物理仿真为 500 Hz。
501 个控制样本后按原基准停止施力，自由演化 500 个物理步。

两次控制期间都有 398 个采样时刻检测到布料与挂架接触。扰动侧六维轨迹距离
峰值为 2.838，最终为 0.01056；外力停止后约 2.34 s 降至 0.05 以下并至少保持
31 个控制样本。这里距离均为 **DEDO 仿真尺度**，不能当作米或毫米。
0.05 是恢复诊断阈值，不是基准成功判据；它没有用于修改控制或筛选结果。

**两次最终挂布判据均未通过**，官方预设直接回放在本机的 3.1.7 和 3.2.7 下也
未通过。当前展示的是软体接触环境中的夹持点流场执行与扰动恢复，不能据此声称
挂布任务成功或布料整体状态恢复；流场输入只有夹持点位置，并不含全布料形状。
这里是单条公开轨迹的可行性测试，不是学习策略的统计评估。

`rollout.npz` 保存全部网格顶点、夹持点、速度指令、接触计数、奖励和自由演化标志；
`protocol.json` 与源快照记录来源、参数和 SHA256。`--compare` 在单点调用布局下
重算每条流场指令（逐元素完全一致），另用 20,001 点密集曲线独立检查距离，并核对
共同参考、扰动前状态、施力区间和源／资源哈希。图、MP4 和 `audit.json` 保存在
`published/`；视频尾部的 `released` 表示已停止控制，恢复曲线只统计控制阶段。
开发阶段的版本对照、初始渲染和一次批量／单点浮点差异造成的核验失败均保留，
未覆盖物理结果，也未按分数重跑。

### 接触前双点扰动：24 条件统计（2026-09-28）

```bash
# 固定网格完整执行；低分及提前终止不重跑。
../../.venvs/safelibero/bin/python run/rebuttal/dedo_flow.py \
  --sweep --output outputs/dedo_precontact_new
# 只从保存的视频重绘两行五帧，不重跑仿真。
../../.venvs/safelibero/bin/python run/rebuttal/dedo_flow.py \
  --sequence outputs/dedo_precontact_20260928 --output /tmp/unused
```

运行目录 `outputs/dedo_precontact_20260928` 的 `manifest.json` 在执行前固定了六个
方向（±x/±y/±z）×两档力（12/16 N）×两个起始时刻（0.4/0.8 s），共 24 个条件。
每次两个夹持点受到相同方向、相同大小的外力，持续 5 个控制周期（0.08 s）。
采用整数控制周期定义开关，避免浮点端点比较多施力一步。其余公开轨迹、流场、
原控制器、仿真和释放判据保持不变。未重训或修改共享模块。

全部 24 次施力前及施力期间都未接触任何刚体。19/24 在接触挂架前满足恢复条件，
并到达下降动作的终点；5/24 触发原环境的提前终止。最终基准挂布判定为 **0/24**。
恢复条件是六维夹持点到样条的距离 <0.05 仿真单位，连续维持 11 个控制更新，
且在首次接触前完成；终点条件为双点联合误差 <0.1 仿真单位。这里统计固定扰动
条件，不把它当作独立随机场景的成功率估计。完整 CSV/JSON 包含全部失败。

提前终止：`x_pos_f16_t040`、`y_pos_f16_t040`、`z_neg_f12_t040`、
`z_neg_f16_t040`、`z_neg_f16_t080`。没有以新的结果替换它们。
原低层控制器使用局部坐标施力，额外坐标系诊断的力方向余弦最低仍为 0.99999996；
该诊断未复现所选失败，因此不足以将失败归因于坐标系，也未据此修改控制器。
诊断目录 `../dedo_controller_probe_20260928` 不计入 24 条件统计。

图的两条序列在运行前指定为 `x_pos_f16_t040` 和 `z_pos_f16_t040`，没有改选成功例。
第一行短暂恢复后提前终止；第二行恢复后继续下降。`figure_frames.json` 给出全部
帧索引，`disturbance_sequence.png/pdf` 为两行五帧图。初始绘图因预设案例未到达
下降阶段而报错，统计已经完整；绘图改为明确显示 `Early stop`，未改变 raw 数据。

`independent_audit.json` 核对 24 个条件、各 5 步外力、无施力前接触、CSV/JSON 数量、
提前终止、未改动的基准判据、raw SHA256 和唯一执行源码快照。每条流场指令在
单点调用下逐元素重放一致，并用密集曲线独立检查距离。各次 reset 的微小数值差异
使起点拟合权重的最大差异为 9.54e-6，不能写成跨试验逐位相同的物理初态。


## R1 Other 1：SoftGym 折布与双点扰动（2026-09-28）

入口只有 `softgym_flow.py`，包含执行、独立 raw 核验和时序绘图。正式输出为
`outputs/softgym_fold_20260928/`；原 DEDO 挂布的代码、238 项输出／图文件及负结果原样保留，
不改变挂布成功标准。探索阶段的铺展任务存在自接触／展开不足，未进入正式统计；
这些预检以及首次折布过早放手和日志类型错误都已归档，不能混入正式数据。

采用公开 [SoftGym ClothFold](https://github.com/Xingyu-Lin/softgym) 原环境，commit 见 manifest。
原始默认布料尺寸、seed 0 的原始随机旋转、刚度、重力、摩擦、地面和自碰撞模型均保留；
固定一个配置，使用两个半径 0.015 的原生 picker。初始时将 picker 放在两个相邻角点，
通过原始拾取接口夹持；执行期间不设置布料粒子位置来修正结果。
将该边绕中线翻到另一边的半圆路径拟合为 12 段、6 维二次样条，固定起终点与终端零切向。
每次从实际两点位置调用 release 的 `dynamical_system_single_step`，λ=5，6D 最大速度 0.15，
100 Hz；不训练、不重新规划，不反馈布料形状。共享流场、其他实验和 upstream 源码均未修改。

正式方案事先固定：1 次无扰动 + 24 次双点扰动，六个正负坐标方向 × 0.04/0.08 位移 ×
1.5/2.5 s 开始时刻，持续 0.2 s；两点同时加相同位移增量。**原 picker 是运动学接口，
这里是执行位移扰动，不是牛顿单位的外力。** 每个物理步都执行原软体求解，实际位移与
请求位移独立核对以检查裁剪。15 s 松开两点，继续原物理 2 s；松开后全部粒子恢复有限质量，
picker 球体仍留在原处，不声称它们已经撤离接触区域。25 次全部重新执行，不拼接预检，不按分数重跑。

统计保持原 `ClothFoldEnv._get_info()`：两半对应粒子平均距离，加 1.2 倍固定半边位移惩罚，
并按原初始误差归一化；分数越高越好，不新增或放宽二元任务成功阈值。
恢复指 6D 夹持点到原曲线距离连续 10 个更新小于 0.015，并在原路径中点（开始下降）之前完成。
恢复次数与任务分数分别报告，不把回到夹持点路径等同于全部布料状态恢复。

复现（本机隔离环境，不影响原训练环境）：

```bash
# 在 release/spline_policy 下；新的输出目录必须不存在。
bash outputs/softgym_fold_20260928/environment.sh run/rebuttal/softgym_flow.py --output outputs/softgym_fold_new --suite
bash outputs/softgym_fold_20260928/environment.sh run/rebuttal/softgym_flow.py --output outputs/softgym_fold_new --audit
bash outputs/softgym_fold_20260928/environment.sh run/rebuttal/softgym_flow.py --output outputs/softgym_fold_new --plot
```

SoftGym 和编译产物位于忽略的 `data_local/softgym`，兼容依赖在 `data_local/softgym_deps`。
原 FleX 链接隔离的官方 CUDA 9.2.148 runtime；glibc 旧符号 `__powf_finite` 转发给 `powf`。
NumPy 1.23.5、pybind11 2.10.4 也在隔离 target 中；没有修改系统驱动或共享 venv。
源码版本、二进制／依赖 hash、环境入口和构建／失败日志都在正式输出目录。

`rollout.npz` 保存每步位置、实际请求动作、扰动、最近距离／相位、官方任务指标、拾取状态，
以及每 10 步的完整粒子位置与逆质量。`--audit` 从粒子重算折叠／固定半边误差，核对全部动作、
双点扰动、有限数值、释放、恢复计数；50,001 点独立近邻核验连续曲线距离，并验证相同的恢复时刻。
图为预先指定的 x+、y+ 大幅扰动、2.5 s 开始两个案例，各五帧，完全取自同轮真实物理录像。
紫色为实际 picker 轨迹，红箭头表示扰动方向，未改变布料形态或平滑实际路径。
单一配置的 24 条件不是 24 个独立随机种子，也不是训练策略的全基准成功率。

正式结果：25/25 执行完整、无技术失败或重试；24/24 扰动在下降前恢复。
官方归一化分数 **0.846613 ± 0.058546**（sample SD），无扰动 **0.878447**，
范围 **0.610470–0.878957**。最差 x−/0.08/1.5 s 保留：夹持点恢复后仍有布料残余形变／固定半边移动。
对应粒子折叠误差 0.039323 ± 0.007194、固定半边位移 0.005578 ± 0.008776；数值均为仿真单位。
所有初始夹持位置和拟合参数逐元素一致，全部动作 CPU 回放最大误差 6.93e-8。
25 组粒子数据逐帧重算官方指标通过；所有夹持点执行增量匹配动作，无限位裁剪，
放手后 5510 个粒子全部恢复有限质量。没有用视觉上的折好与否替代公开指标。

独立数值核验发现共享 float32 SDF 在近零距离处与双精度三次极值解有约 0.00025 的舍入误差；
50,001 点独立距离重新得到的全部 25 个恢复时刻完全一致，无需更改流场或恢复阈值。
逐帧 float32 归约顺序用于精确复算官方指标；最初批量归约的审计差异和审计改动已留档。
冻结执行源码与完成后的审计源码分别保存，正式 raw、场景、轨迹、动作、指标没有改写。
`audit.json`、`independent_distance.json`、`summary.json`、`results.csv` 和 `figure_frames.json` 给出证据。
原 DEDO 的 238 项文件 hash 全部一致。


### R1 Other 1：加大扰动并显示真实流场（2026-09-28）

本轮正式输出 `spline_policy/outputs/softgym_fold_stronger_20260928/`，幅度由 0.04/0.08
改为 0.08/0.12，原六方向、两时刻、流场、轨迹、物理与评价标准保持不变。
预检两展示案例可恢复且无动作裁剪后固定全部条件；正式 25 次全新执行，未筛选或拼接。
24/24 夹持点在下降前恢复；官方分数 0.758725±0.156391（sample SD），
无扰动 0.878469，范围 0.229845–0.879354。最差 x−/0.12/1.5 s 保留，
对应折叠误差 0.115348、固定半边位移 0.096415：夹持点恢复不等于布料状态完全恢复。
没有单独 baseline 控制器；无扰动执行只作 nominal 对照。全部低分保留，无重跑或技术失败。

仍用事前指定的 x+、y+ /0.12/2.5 s 展示。蓝线从真实六维流场联合积分，再分别投影到
两个夹持点的三维空间；两点同时移动，不冻结另一点，也不将它描述为布料粒子的流场。
每个场景的六个初始偏移只用于显示附近流线，不是额外实验；绘图积分步长 0.001、360 步，
同 λ=5 和共享流场，空间映射与仿真相机一致。紫线为未平滑的真实执行路径。
`flow_streamlines.npz` 保存查询路径／方向，独立回算全部 8640 个方向和积分更新。
颜色／字体沿用论文，流场仅叠加在扰动与恢复两列，保留两行五帧结构。

正文删除阈值、单位和条件组合的详细脚注，只简述不同扰动及结果；没有将分数下降写成
与 nominal 相当或全部任务成功。原小扰动 150 项归档文件 hash 全部一致，旧图和回复另存
本轮 `paper_before/`。本次只更新现有 `softgym_flow.py`；正式执行冻结快照与最终绘图代码
分别保留，AST 核对只有绘图函数变化。共享源码、旧结果和其他实验均未修改。


### 2026-09-28：SoftGym 完整平面流场图

按用户要求将正文顺序改为：无扰动 normalized score **0.878**，扰动后
**0.759±0.156**（mean±sample SD）。统计与全部 raw 不变。

图从少量局部流线改为完整二维网格的场：在前方夹持点的 XY 平面采样 161×121 个位置，
两个夹持点作相同 XY 平移，固定当前 Z 坐标及两点间距；用原六维控制器计算每个查询，
将方向正交投影到这个共同平移平面，再以原相机透视映射绘制浅蓝连续流线。
这是原场的二维截面／投影，不是新增控制器，也不代表布料所有粒子的速度场。
图注只简写 planar flow field；保持两行五帧、实际紫色轨迹与同一组选帧。

新交付与网格数据在 `softgym_fold_stronger_20260928/planar_visualization/`。
逐点独立重算流场混合公式，四个网格各 19,481 点，最大方向误差 1.20e-7；
二维投影、平面位置和未改变的帧记录核验通过。之前 138 个归档文件 hash 全部一致，
执行、任务指标、恢复判据和审计函数 AST 均未改变；仅修改现有入口的绘图部分。
复绘可用 `--plot --figure-dir outputs/softgym_fold_stronger_20260928/planar_visualization`，
配合原 `--output outputs/softgym_fold_stronger_20260928`。不重跑物理，不修改共享流场。


### 2026-09-28：SoftGym 全幅面场及轨迹样式

按用户要求，轨迹采用论文 #F1454F → #E388B0 → #C0B4FF → #7296FF 渐变，
沿当前显示的原始轨迹时间顺序着色；中心线不平滑或修改。彩色线宽从 5 px 改为
12.5 px（2.5 倍），高分辨率圆角描边避免短线段接缝。流线和箭头宽度各为之前 2 倍。

十个画面均显示面场。用原相机光线反投影求 XY 平面覆盖范围，覆盖原裁剪图四角并加
少量余量；固定原相机和画面，未扩大或缩小物理轨迹。网格由原控制器计算，
扩展区域是数学流场的可视化，不表示该区域均在执行器工作范围内；Released 帧显示
相同固定轨迹定义的场，不表示松手后仍施加控制。原仿真和统计不重跑、不修改。

全幅可视化保存在 `softgym_fold_stronger_20260928/fullframe_visualization/`，
10×19,481 个场查询及二维投影公式核验通过，原始归档 155 文件 hash 保持一致，
相机、选帧、执行函数和统计均未改变。caption 将 purple 更新为 gradient curves。


### 2026-09-28：原生 streamplot 与全局轨迹

按用户要求替换手动绘制的线段／三角箭头。现在直接使用 Matplotlib 原生 `streamplot`，
参数参照 `spline_policy_demo.ipynb` 与 `plot_projection_comparison.py`：density=0.6、
arrowsize=1.2、minlength=0.15、maxlength=2.0，保留默认的流线相遇终止行为。
不再强制延长相邻流线或重画箭头；高分辨率原生抗锯齿层叠加到原仿真帧。

全局面取自参考折叠运动的固定平面，六维原点／基向量由两点起终位置确定，
不是随当前状态移动的 XY 截面。把屏幕规则网格反投影到该平面，查询同一个六维
控制器，再用平面正交投影和相机 Jacobian 转为屏幕方向。十帧的场与完整参考轨迹
逐元素相同，始终覆盖整个画面；参考曲线在平面内的最大残差 9.91e-8。

完整样条参考轨迹始终以论文渐变色显示，线宽 12.5 px；实际轨迹改为从起点累计到
当前帧的紫色路径（6 px），不再截取最近 160 步。caption 区分 reference 与 execution，
没有把尚未执行的参考部分画成已经执行的轨迹。此前询问了全局轨迹显示方式，
未收到答复后按已说明的“完整参考＋累计执行”方式交付。

本次归档 `softgym_fold_stronger_20260928/global_streamplot/`；公式重算误差 1.20e-7，
相机 Jacobian 有限差分核验最大差异 2.04e-4，十帧全局场／曲线和实际选帧
核验通过。原 171 个文件 hash 保持一致，原执行、指标、权重、控制器均未修改。


### 2026-09-28：流线加密与扰动轨迹配色

原生 streamplot 的 density 从 0.6 增至 1.0。全局参考保留论文红蓝渐变，
实际累计执行轨迹采用原 PPT 扰动对比图的浓紫色 `#8A2BE2`（slide 28），
使偏离与恢复段更清楚。归档在 `softgym_fold_stronger_20260928/dense_disturbance_colors/`。
新旧场数组、全局路径和选帧逐元素一致，187 个原归档文件 hash 不变；
未重跑仿真，指标与 response 正文保持不变。


### 2026-09-28：前／后半段扰动示例

图中移除额外 disturbance 箭头，保留流场自身箭头；A 沿用原 lateral case
`x_pos_120_250`（onset phase 0.2833），B 新跑 `y_pos_120_600`
（phase 0.6860），幅度仍为 0.12。B onset=600 在运行前由原 nominal phase 选定，
未按结果挑选或重试。两点同时受扰；step 693 起满足连续十步路径恢复标准，
最终原 SoftGym score=0.7383746，step 1500 松手。B 在下降后受扰，
不能称其 before-lowering recovery。首列标题改为 Before，五帧时序均已核验。

新案例、复用 A/nominal 的逐字节副本和运行脚本归档于
`spline_policy/outputs/softgym_fold_early_late_20260928/`。原 24-trial 统计不混入此
示例，response 明确统计属于 lifting-perturbation study；新目录 summary 仅描述
示例集，不用于正文均值。粒子指标独立重算、命令重放、恢复距离、首状态／权重
一致性核验通过；全局流场与参考未变，201 项历史归档 hash 保持一致。


### 2026-09-28：两行均采用向上扰动

A 替换为原 24-trial 数据中的 `y_pos_120_250`，B 保留 `y_pos_120_600`。
两者均为两抓点同时向上 0.12 的位移扰动，onset phase 分别为 0.2833／0.6860，
恢复 step 为 336／693。沿用原 raw 和视频，不重跑仿真，原统计不变。
图中不画额外扰动箭头，视频标签简化为 A／B，caption 标明 upward。
归档：`softgym_fold_upward_pair_20260928/`；原始副本逐字节一致，242 项保留 hash 通过。


### 2026-09-28：只显示流场对应抓点的轨迹

按要求隐藏上方抓点的参考／实际轨迹，仅显示 `field["front"]` 对应的下方轨迹。
两点物理执行与扰动不变，场、原始轨迹数组、选帧逐元素相同；仅减少叠加曲线。
新图归档于 `softgym_fold_upward_pair_20260928/front_only/`，原图和数据保留。
