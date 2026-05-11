# MM-Flow Research Track

## 项目目标

暂定题目：

> MM-Flow: Plan-to-Plan Generative Replanning for Safe Whole-Body Mobile Manipulation

核心问题：

> 在 receding horizon 框架下，能否利用上一周期的 whole-body trajectory 作为 source / warm start，快速生成下一周期的安全可行轨迹，同时满足底盘-机械臂耦合、全身避障和动态可行性？

这个项目的核心 claim 不应该是“用历史初始化 diffusion”。这个说法太宽，已有相关工作不少。更合适的 claim 是：

> 面向移动机械臂耦合全身轨迹的 plan-to-plan generative replanning。

项目需要始终围绕下面几个点：

- previous whole-body plan -> next whole-body plan
- base-arm coupling
- whole-body collision safety
- dynamic closed-loop replanning
- limited local perception

如果最后 novelty 只能说成“我们用了上一帧轨迹”，那是不够的。必须落到移动机械臂的 whole-body 闭环重规划问题上。

## Phase 0: 文献和问题边界

目标：明确 novelty，避免和 Diffusion Policy、A2A、Streaming Flow Policy、PTDM 等工作过于重合。

需要整理的相关工作：

- Diffusion Policy / A2A / Streaming Flow Policy
  - 历史动作输入
  - receding-horizon action chunks
  - 低延迟 action generation
- MPD / M2Diffuser / PTDM
  - diffusion trajectory generation
  - trajectory prior
  - mobile manipulation trajectory generation
- FlowMP / Safe Flow Matching
  - flow matching for motion planning
  - safety-aware flow matching
- Whole-body reactive control
  - link-specific safety
  - free-region constraints
  - AL-DDP 或类似局部优化 / 投影方法

预期产出：

- `papers/related_work.md`
- 一张对比表：

```text
method | robot type | source distribution | output | safety | closed-loop | mobile manipulator
```

判断 novelty 的核心标准：

```text
previous whole-body plan -> next whole-body plan
with base-arm coupling + whole-body safety + dynamic replanning
```

## Phase 1: Toy Planar Mobile Manipulator

目标：先在简单可控的系统里验证核心假设，不要一开始就上复杂 URDF 或真实机器人栈。

机器人模型：

```text
base: x, y, theta
arm: q1, q2
state: [x, y, theta, q1, q2]
```

任务设置：

- 2D start-to-goal planning。
- 末端执行器到达目标点。
- 环境包含圆形或矩形障碍物。
- 底盘 body 和机械臂 links 都需要 collision-free。
- 每次规划长度为 `T` 的 horizon，只执行前 `k` 步，然后重新规划。

输入：

```text
shifted previous trajectory tau_{t-1}
current state x_t
goal g
obstacle representation o_t
```

输出：

```text
next horizon trajectory tau_t
```

初始 baselines：

```text
1. noise-to-trajectory diffusion / flow
2. shifted previous trajectory + gradient optimization
3. P2P: previous trajectory -> next trajectory
```

评价指标：

```text
success rate
collision rate
inference time
smoothness
trajectory consistency
number of refinement steps
```

预期产出：

- 一个可视化 demo：动态障碍出现时，上一轮轨迹被快速修正。
- 第一组核心实验图。
- 初步判断 plan-to-plan generation 是否真的有优势。

## Phase 2: Plan-to-Plan Flow / Diffusion Model

目标：确定生成模型形式。

建议先做 flow matching，而不是完整 diffusion。原因是目标应用是实时 replanning，flow matching 更容易往 one-step / few-step inference 方向推进。

候选 formulation：

```text
source z0:
  encoded shifted previous trajectory

condition c:
  current state
  goal
  obstacle features
  tracking error

target z1:
  expert next trajectory
```

训练目标：

```text
learn v_theta(z_t, t, c)
where z_t = (1 - t) z0 + t z1
```

推理过程：

```text
tau_t = integrate flow from the previous-plan source to the corrected plan
```

重要 ablations：

- Gaussian noise source vs previous-plan source。
- last action input vs full previous trajectory input。
- 无 obstacle condition vs 有 obstacle condition。
- 无 safety refinement vs 有 safety refinement。
- one-step / two-step / multi-step inference。

预期产出：

- 模型代码。
- toy environment 上的完整实验。
- 图表证明 plan-to-plan transport 比 noise-to-plan 更短、更稳定、更快。

## Phase 3: Safety Layer

目标：不能只靠模型“学会安全”，需要有显式 safety mechanism。

初始 cost 设计：

```text
obstacle collision cost
self-collision cost if needed
joint limit cost
smoothness cost
goal cost
```

Safety refinement 选项：

```text
A. gradient-based projection
B. lightweight trajectory optimization
```

后续可以扩展：

```text
link-specific free regions
CBF-style safety projection
AL-DDP-style local refinement
```

关键假设：

> Plan-to-plan source 会让 safety projection 更容易，因为生成结果本来就靠近上一轮可行计划，而不是从完全随机的 noise 开始。

实验内容：

- refinement 前的 collision rate。
- refinement 后的 collision rate。
- projection / refinement 需要多少步。
- 在相同 refinement budget 下，与 noise-based generation 对比。

## Phase 4: 简化 3D Mobile Manipulator

目标：从 toy validation 过渡到更真实的问题，但仍然保持系统可控。

机器人模型：

```text
base: x, y, yaw
arm: 4-DoF or simplified 6/7-DoF arm
```

环境：

- cuboids
- cylinders
- floating obstacles
- narrow passages
- dynamic obstacles

轨迹表示候选：

```text
base: x(t), y(t), yaw(t)
or base: arc-length/yaw parameterization
arm: q(t)
```

需要加入的约束：

- kinematic feasibility
- velocity limits
- acceleration limits
- whole-body collision
- receding-horizon closed loop

预期产出：

- 3D simulation results。
- 更接近移动机械臂轨迹规划论文的实验设置。
- 能展示 base-arm coordinated behavior 的例子。

## Phase 5: Full MoMa Model

目标：使用已经复制到项目里的 MoMa 模型资产，基于真实机器人几何做 closed-loop replanning。

主要模型文件：

```text
assets/robots/moma/whole_body_description/urdf/mobile_piper.urdf
```

需要完成：

- 解析 URDF。
- 获取 link geometry。
- 建立 forward kinematics。
- 必要时简化 collision model。
- 定义 whole-body state。

初始 whole-body state 可以先设为：

```text
[x, y, yaw, q1, q2, q3, q4, q5, q6]
```

这一阶段先专注仿真和 closed-loop replanning，不要一开始就接真实控制。

实验场景：

- static cluttered scene
- dynamic obstacle scene
- narrow passage
- end-effector target tracking
- base detour with arm retraction
- base pose adjustment for improved arm reachability

## Phase 6: 论文实验设计

主实验：

```text
MM-Flow vs baselines
```

Baselines：

```text
1. noise-to-plan diffusion / flow
2. previous plan + optimization only
3. primitive-based generation
4. classical local planner / MPC
5. PTDM-style method if feasible to reproduce
```

Metrics：

```text
success rate
collision rate
planning latency
closed-loop replanning frequency
trajectory smoothness
goal error
minimum obstacle distance
optimization refinement steps
```

Ablations：

```text
without previous plan
without action history
without safety projection
without obstacle condition
single-step vs multi-step flow
waypoint output vs spline / polynomial output
```

最重要的图：

- 动态障碍出现后，轨迹如何从上一轮 plan 平滑修正。
- plan-to-plan source-target transport vs noise-target transport。
- 相同 safety refine 步数下，P2P 的成功率是否更高。
- planning latency 对比。
- 展示 whole-body behavior：底盘绕行、手臂收缩、末端跟踪、底盘调整姿态提高可达性。

## 建议时间线

### Week 1-2

- 完成 related-work table。
- 搭 planar mobile manipulator 环境。
- 实现数据生成和可视化。

### Week 3-4

- 实现 plan-to-plan flow matching。
- 跑 toy baselines。
- 测试 P2P 是否优于 noise source。

### Week 5-6

- 加 safety projection。
- 做 dynamic-obstacle closed-loop experiments。
- 产出第一版实验图。

### Week 7-9

- 迁移到 simplified 3D model。
- 加 link collision 和 velocity / acceleration constraints。
- 做 3D ablations。

### Week 10-12

- 导入并使用真实 MoMa URDF。
- 建立 kinematics 和 collision 支持。
- 跑完整 closed-loop simulation demo。

## 当前下一步

不要先训练大模型，也不要一开始就上真实机器人模型。

下一步的具体任务：

```text
1. 搭建 planar mobile manipulator toy environment。
2. 实现 trajectory / state / collision 基础模块。
3. 生成 expert trajectories。
4. 实现 previous-plan shift 和 receding-horizon loop。
```

如果这个最小闭环能跑起来，后面的 flow / diffusion model 才会有清晰的训练目标。

## 当前进展

### 2026-05-11

已完成 Phase 1 的第一版 2D start-to-goal planning demo：

- 搭建了 planar mobile manipulator toy model：
  - base: `x, y, theta`
  - arm: `q1, q2`
  - state: `[x, y, theta, q1, q2]`
- 实现了基础模块：
  - `src/mm_flow/planar/kinematics.py`
  - `src/mm_flow/planar/collision.py`
  - `src/mm_flow/planar/heuristic_planner.py`
  - `src/mm_flow/planar/optimizer.py`
  - `src/mm_flow/planar/visualization.py`
- 实现了 demo 入口：
  - `experiments/planar_start_goal_demo.py`
- 当前默认 planner 是 fast geometric whole-body seed generator。
  - 它不是最终方法。
  - 作用是先把可视化、运动学、碰撞检测和 start-to-goal whole-body trajectory scaffold 跑通。
- 默认 demo 输出：
  - `outputs/planar_start_goal/trajectory.png`
  - `outputs/planar_start_goal/trajectory.npz`
- 当前默认结果：
  - success: `True`
  - terminal error: `0.0000 m`
  - min clearance: `0.2675 m`

验证：

```bash
PYTHONPATH=src pytest -q
```

结果：

```text
1 passed
```

下一步应该做：

```text
1. 把当前 start-to-goal demo 扩展成 receding-horizon loop。
2. 实现 previous-plan shift。
3. 加动态障碍物，让每一步 replan 有实际意义。
4. 保存每一轮 previous plan / current state / next plan，作为后续 P2P flow matching 的数据格式雏形。
```

### 2026-05-11 更新

调整 Phase 1 的 demo 目标：不要只做普通 start-to-goal navigation，而是给机械臂明确的 reaching / grasp-like task。

已新增 sequential reaching demo：

- 文件：
  - `experiments/planar_reach_sequence_demo.py`
- 任务：
  - 末端执行器依次 reach 多个 2D goal points。
  - 底盘和机械臂需要协同移动。
  - 整个过程需要满足 whole-body collision-free。
- 输出：
  - `outputs/planar_reach_sequence/reach_sequence.png`
  - `outputs/planar_reach_sequence/reach_sequence.npz`
  - `outputs/planar_reach_sequence/reach_sequence.gif`
- 当前默认结果：
  - success: `True`
  - goals: `3`
  - trajectory steps: `120`
  - 每个 goal 的 terminal error 都为 `0.0000 m`
  - 最小 segment clearance 约为 `0.1541 m`

这个 demo 更接近后续论文想表达的 mobile manipulation：

```text
不是单纯让底盘导航，也不是让机械臂被动避障；
而是移动底盘和机械臂在同一时间轴上协同，使 end-effector reach 一系列目标点。
```

已补充动画可视化：

- `reach_sequence.gif` 会按时间播放底盘、机械臂、末端轨迹和当前目标点。
- 这个输出用于直观看 base-arm coordination 是否合理。

### 2026-05-11 3D Toy Demo

已开始 Phase 4 的简化 3D 场景，但仍然不直接上真实 URDF。

新增 simplified 3D mobile manipulator：

- base: `x, y, yaw`
- arm: 3-DoF spatial arm
  - `q1`: yaw
  - `q2`: shoulder pitch
  - `q3`: elbow pitch
- state: `[x, y, yaw, q1, q2, q3]`

新增文件：

- `src/mm_flow/three_d/kinematics.py`
- `src/mm_flow/three_d/collision.py`
- `src/mm_flow/three_d/heuristic_planner.py`
- `src/mm_flow/three_d/tasks.py`
- `src/mm_flow/three_d/visualization.py`
- `experiments/three_d_reach_sequence_demo.py`
- `tests/test_three_d_demo.py`

3D demo 任务：

- 末端执行器依次 reach 3 个 3D goal points。
- 底盘在地面平面移动。
- 机械臂在 3D 空间 reach。
- 障碍物包含 spheres 和 cuboids。
- 使用简化 whole-body collision checking。

当前默认结果：

```text
success: True
goals: 3
trajectory steps: 132
goal 1: error=0.0000 m, min_clearance=0.3189 m
goal 2: error=0.0000 m, min_clearance=0.3448 m
goal 3: error=0.0000 m, min_clearance=0.3452 m
```

输出：

- `outputs/three_d_reach_sequence/reach_sequence_3d.png`
- `outputs/three_d_reach_sequence/reach_sequence_3d.npz`
- `outputs/three_d_reach_sequence/reach_sequence_3d.gif`
- `outputs/three_d_reach_sequence/reach_sequence_3d.html`

已补充交互式 3D viewer：

- 文件：
  - `src/mm_flow/three_d/html_viewer.py`
- 输出：
  - `reach_sequence_3d.html`
- 使用方式：
  - 在浏览器中打开 HTML。
  - 鼠标拖动旋转视角。
  - 滚轮缩放。
  - 右键拖动平移。
  - 底部 slider 调整时间步。

备注：

- 本机装了 `rerun`，但当前 Python 3.8 环境导入 rerun SDK 失败，原因是 SDK 类型标注需要更高 Python 版本。
- 因此当前先用 Three.js HTML viewer，避免新增 Python 依赖。

2026-05-11 viewer 修正：

- 修复 Three.js viewer 的坐标系问题：
  - robot/world 坐标是 Z-up: `[x, y, z]`
  - Three.js 默认 Y-up
  - 现在渲染映射为 `[x, z, -y]`
- 底盘数据层面始终满足：
  - `base z = 0`
  - `base center z = 0.16`
- 新增 Matplotlib 交互查看脚本：
  - `scripts/view_three_d_matplotlib.py`
  - 需要本地图形界面和可用 GUI backend。

2026-05-11 播放速度和跳变处理：

- HTML viewer 新增播放速度选择：
  - `0.5x`
  - `0.75x`
  - `1x`
  - `1.5x`
  - `2x`
- 观察到的中段“机械臂跳变”主要来自分段拼接时的 base yaw 突变，而不是机械臂关节本身的大幅跳变。
- 保留原始 planner 输出，不直接篡改训练/评估轨迹。
- 对 PNG/GIF/HTML 可视化轨迹做 densify 插帧：
  - 原始 3D trajectory: `132` frames
  - HTML visualization: `195` frames
  - 这样动画播放更平滑，同时 `reach_sequence_3d.npz` 仍保存原始轨迹。

### 2026-05-11 只保留 3D 场景

按当前研究方向，已删除 2D 场景逻辑，项目后续只维护 3D simplified mobile manipulator scaffold。

已删除：

- `src/mm_flow/planar/`
- `experiments/planar_start_goal_demo.py`
- `experiments/planar_reach_sequence_demo.py`
- `tests/test_planar_demo.py`
- `outputs/planar_start_goal/`
- `outputs/planar_reach_sequence/`

当前保留：

- `src/mm_flow/three_d/`
- `experiments/three_d_reach_sequence_demo.py`
- `tests/test_three_d_demo.py`
- `outputs/three_d_reach_sequence/`

3D 障碍物系统已扩展：

- `SphereObstacle`
- `CuboidObstacle`
- `CylinderObstacle`
- cube 作为特殊 cuboid：`size=(s, s, s)`

随机障碍物生成：

- 文件：`src/mm_flow/three_d/tasks.py`
- 函数：`build_reach_sequence_scene_3d(seed, random_obstacle_count)`
- 支持：
  - sphere
  - cube
  - cuboid / rectangular box
  - vertical cylinder
  - grounded obstacles
  - floating obstacles
- demo 参数：

```bash
PYTHONPATH=src python3 experiments/three_d_reach_sequence_demo.py --seed 11 --obstacle-count 16
```

注意：

- 当前 heuristic planner 只负责生成 scaffold 和可视化，不保证复杂随机障碍物下一定 collision-free。
- 这符合当前阶段目标：先构建 3D 随机场景和任务接口，后续再用优化 / flow matching / diffusion 改善安全性。

### 2026-05-11 RRT-Connect Expert Planner

新增 whole-body RRT-Connect planner，用于替代 heuristic 生成 collision-free 轨迹。

文件：

- `src/mm_flow/three_d/rrt_connect.py`

规划状态：

```text
[x, y, yaw, q1, q2, q3]
```

核心逻辑：

- 双向 RRT-Connect。
- goal tree 由多个 IK goal states 初始化。
- 状态有效性由 `state_clearance` 判断。
- edge validity 通过沿边插值逐点 collision check。
- 成功后做 shortcut smoothing。

Demo 默认 planner 已切换为：

```bash
PYTHONPATH=src python3 experiments/three_d_reach_sequence_demo.py --planner rrt_connect
```

保留 heuristic 作为 scaffold：

```bash
PYTHONPATH=src python3 experiments/three_d_reach_sequence_demo.py --planner heuristic
```

默认随机场景当前结果：

```text
planner: rrt_connect
success: True
goals: 3
trajectory steps: 160
goal 1: error=0.0163 m, min_clearance=0.0015 m
goal 2: error=0.0000 m, min_clearance=0.0150 m
goal 3: error=0.0000 m, min_clearance=0.1668 m
```

备注：

- RRT 当前能生成 collision-free path，但 clearance 可能贴近 0。
- 下一步可加 safety margin、path smoothing、trajectory optimization，避免路径擦障碍。

### 2026-05-11 HTML Runtime HUD

Three.js HTML viewer 已加入紧凑运行状态面板。

文件：

- `src/mm_flow/three_d/html_viewer.py`
- `experiments/three_d_reach_sequence_demo.py`

HTML 中现在会显示：

- planner / algorithm，例如 `rrt_connect`
- success status
- seed 和 obstacle count
- 当前 frame 和当前 target goal
- base pose: `x, y, yaw`
- arm joints: `q1, q2, q3`
- 当前 end-effector 到目标点的误差
- 当前 clearance 和全局最小 clearance

设计原则：

- 信息面板固定在右上角，半透明、小尺寸。
- 只显示调试 planning 行为所需的实时信息。
- 不做大面积日志输出，避免遮挡 3D 场景主体。

### 2026-05-11 RRT-Connect Failure Validation

新增一个可复现的 RRT-Connect 失败验证脚本：

- `experiments/rrt_connect_failure_validation.py`

验证目的：

```text
同一个 whole-body planning problem 本身是可解的；
但在较紧的在线重规划预算下，RRT-Connect 可能无法及时找到可行轨迹。
```

默认 case：

```bash
PYTHONPATH=src python3 experiments/rrt_connect_failure_validation.py
```

当前结果：

```text
scene: seed=0, obstacles=14, goal=1
clearance margin: 0.020 m
low budget:
  max_iterations=250
  goal_samples=20
  success=False
  message=max iterations reached
  time=1.373 s
  terminal error=1.8078 m

reference budget:
  max_iterations=8000
  goal_samples=160
  success=True
  message=connected
  time=12.797 s
  terminal error=0.0000 m
  min clearance=0.0213 m
```

输出：

- `outputs/rrt_connect_failure_validation/summary.json`
- `outputs/rrt_connect_failure_validation/reference_solution.npz`
- `outputs/rrt_connect_failure_validation/reference_solution.html`

这个实验说明：

```text
RRT-Connect 可以作为 expert generator / baseline；
但如果每次新观测后都从当前 state 重新搜索，
在线 replanning 的 latency 和 success 都不稳定。
```

这正是后续引入 receding horizon + plan-to-plan flow / diffusion 的动机：

```text
不要每次从零搜索；
而是利用 shifted previous whole-body plan，
快速生成 next corrected whole-body plan。
```

### 2026-05-11 MoMa/Piper URDF Kinematics + RRT-Connect

已开始把 simplified 3D model 替换为真实移动机械臂模型。

新增：

- `src/mm_flow/three_d/moma_kinematics.py`
- `experiments/moma_rrt_connect_demo.py`

当前使用的模型文件：

- `assets/robots/moma/whole_body_description/urdf/mobile_piper_9dof.urdf`

当前状态定义：

```text
[base_x, base_y, base_yaw, joint1, joint2, joint3, joint4, joint5, joint6]
```

也就是 9D whole-body state：

```text
base: x, y, yaw
arm: 6-DoF Piper joints
```

实现方式：

- 从 URDF 读取 `joint1` 到 `joint6` 的 origin / axis / limits。
- 用 forward kinematics 得到 arm joint chain 和 end-effector point。
- 用 `scipy.optimize.least_squares` 做 position IK，用于 RRT-Connect goal tree sampling。
- RRT-Connect 已改成支持任意 arm joint 维度，而不是只支持 3-DoF toy arm。
- HTML viewer 已改成按 `points` 绘制完整关节链。

当前 collision model：

```text
base: conservative sphere
arm: line segments + link radius capsule approximation
obstacles: sphere / cuboid / cylinder
```

注意：

- 这一步还不是完整 URDF mesh collision。
- 目前目标是先跑通真实模型的 kinematics + whole-body RRT-Connect。
- 后续需要把 collision model 从 capsule approximation 升级到 link-specific geometry 或 mesh/SDF/FCL。
- 2026-05-11 进一步修正：
  - 模型来源明确改为 `/home/yifu/remani_ws/MOMA_ws/src/tools/whole_body_description/urdf/mobile_piper_9dof.urdf`
  - HTML viewer 现在会复制并加载 MOMA_ws 中的真实 mesh：
    - `FW-mini.STL`
    - `base_link.dae`
    - `link1.STL` 到 `link8.STL`
    - `gripper_base.STL`
  - viewer 不再只显示简化线段模型；线段链只作为内部碰撞和调试 scaffold。
  - 当前 RRT collision 仍是 capsule/segment approximation，还不是 mesh collision。

默认运行：

```bash
PYTHONPATH=src python3 experiments/moma_rrt_connect_demo.py
```

当前结果：

```text
robot state dim: 9
goals: 3
success: True
trajectory steps: 87
goal 1: error=0.0025 m, clearance=0.1831 m
goal 2: error=0.0023 m, clearance=0.1831 m
goal 3: error=0.0053 m, clearance=0.1308 m
```

输出：

- `outputs/moma_rrt_connect/moma_rrt_connect.npz`
- `outputs/moma_rrt_connect/moma_rrt_connect.html`
- `outputs/moma_rrt_connect/moma_meshes/`

由于 HTML 需要加载本地 mesh assets，建议用本地静态服务器打开：

```bash
cd /home/yifu/mm-flow
python3 -m http.server 8765 --directory outputs/moma_rrt_connect
```

然后浏览器访问：

```text
http://localhost:8765/moma_rrt_connect.html
```

2026-05-11 viewer frame 标注更新：

- 默认隐藏底盘 heading 蓝线，避免误认为机器人部件。
- 标出以下 URDF link frames 的 xyz axes：
  - `mobile_base` -> `base`
  - `link1` 到 `link6` -> `j1` 到 `j6`
  - `gripper_base` -> `ee`
- 每个 frame origin 都会显示历史运动轨迹。
- `base`, `j1` 到 `j6`, `ee` 的 frame origin 轨迹使用不同颜色。
- 旧的简化 base/ee trace 在真实 mesh viewer 中默认隐藏，避免和 URDF frame trace 混淆。
- 播放到最后一帧后自动停止，不再循环播放。

### 2026-05-11 Link-Specific Collision Proxy

按照 whole-body safety 论文里的思路，先实现 link-specific geometric proxy，而不是直接使用 full mesh collision。

当前新增：

- 每个有 visual geometry 的 link 都有自己的 collision proxy：
  - `mobile_base`: box
  - `aluminum_connector`: box
  - `base_link`: box
  - `link1` 到 `link6`: box
  - `gripper_base`: box
  - `link7`, `link8`: box
- HTML viewer 中用半透明橙色显示这些 proxy，叠加在真实 mesh 上。
- proxy 现在从真实 visual mesh / URDF box 自动计算 link-frame AABB，而不是手工写 capsule/sphere 参数。
- DAE mesh 解析只读取 position array，避免把 normal array 当成顶点导致异常大 box。

当前状态：

```text
- 已建立每个 link 的 proxy。
- 已在 viewer 中可视化 proxy，并提供 proxy 显示开关。
- RRT-Connect 的 collision checking 已切换到这些 link-specific proxies。
```

实现细节：

```text
- MoMa/Piper robot 走 export_collision_proxies + link_transforms。
- simple 3D robot 仍保留旧的 base sphere + arm segment collision checking。
- RRT 内部使用 is_state_collision_free 做快速早停碰撞判定。
- state_clearance 仍用于结果统计和 HTML HUD 的 clearance 显示。
- DAE proxy 生成会过滤极小碎片 geometry，避免 base_link proxy 被不重要的小 geometry 拉长。
- MoMa RRT demo 默认障碍物数量从 14 增加到 24，用于更明显地测试 whole-body proxy collision checking。
```

验证：

```bash
PYTHONPATH=src pytest -q
```

结果：

```text
3 passed
```

24 障碍物 MoMa demo 验证：

```bash
PYTHONPATH=src python3 experiments/moma_rrt_connect_demo.py --max-iterations 1800 --goal-samples 80
```

结果：

```text
success: True
goal 1: error=0.0167 m, clearance=0.0196 m, iterations=42
goal 2: error=0.0886 m, clearance=0.3000 m, iterations=1
goal 3: error=0.0027 m, clearance=0.1780 m, iterations=1
```

2026-05-11 更新：

- 随机场景不再刻意避开 start-goal corridor。
- 每段 start-to-goal / goal-to-goal corridor 附近会优先放置障碍物，让 RRT-Connect 必须表现出绕障，而不是只在空旷走廊中移动。

### 2026-05-11 Planner Registry

当前 `RRT-Connect` 已从 demo 逻辑中拆出来，作为一个可选 planner 保存：

```bash
PYTHONPATH=src python3 experiments/moma_rrt_connect_demo.py --planner rrt_connect
```

新增模块：

```text
src/mm_flow/three_d/planners.py
```

当前 registry 里只有：

```text
rrt_connect -> RRT-Connect whole-body baseline
```

后续加入新 planner 时，优先在 registry 里注册，而不是直接改 demo 主流程。这样同一个 scene / robot / obstacle / visualization 可以复用。

同时对 REMANI-Planner 做了本地代码确认：

- REMANI 的主要前端不是简单的 RRT-Connect。
- 底盘部分主要是 `KinoAstar`，使用 Dubins / kinodynamic-style search，考虑 yaw、forward/backward、singularity 等移动底盘运动约束。
- 机械臂/全身补全里有 `SampleMani` 和 `RrtPlanning`，属于 RRT-style / 双树采样搜索，并带 rewiring / smoothing 逻辑，但代码命名和实现都不是标准教科书版 `RRT-Connect`。
- 后端是 `PolyTrajOptimizer`，用多项式轨迹 + L-BFGS 做 smoothness、obstacle、self-collision、feasibility 等优化。

因此后续对比时可以把 REMANI 表述为：

```text
kinodynamic front-end + manipulator RRT-style search + polynomial trajectory optimization + online replanning
```

而不是简单说成 RRT 或 RRT-Connect。
- 仍保留 start 和 goal 附近保护区，避免直接把问题变成不可解。

走廊障碍版本验证：

```bash
PYTHONPATH=src python3 experiments/moma_rrt_connect_demo.py --max-iterations 3500 --goal-samples 120
```

结果：

```text
success: True
goal 1: error=0.0004 m, clearance=0.0117 m, iterations=61
goal 2: error=0.0407 m, clearance=0.0124 m, iterations=151
goal 3: error=0.0008 m, clearance=0.0530 m, iterations=12
```
