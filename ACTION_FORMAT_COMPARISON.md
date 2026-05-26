# OXE → LeRobot 动作转换对比：我们的格式 vs OpenVLA 格式

两套动作转换共用同一套图像 / state / language 抽取，**只有 action 通道不同**，便于做对照实验。

- **我们的格式（`--action-format ours`，输出 `data/lerobot/`）**：把所有数据集统一成 7D 物理 EEF delta `[Δx, Δy, Δz, rotvec_x, rotvec_y, rotvec_z, gripper]`。位置单位 m，旋转为**轴角**(rad)，绝大多数数据集的 delta 由 **state 的相邻 EE 位姿之差**导出（四元数相对旋转，避免欧拉 ±π / 万向锁），并做单位归一化（mm→m、deg→rad），同时过滤退化帧。**gripper 统一归一化到 `[0,1]`（+1=张开 / 0=闭合），与 OpenVLA 格式采用完全相同的每数据集约定**，因此两套格式的 gripper 语义一致、只在运动通道上有差异。
- **OpenVLA 格式（`--action-format openvla`，输出 `data/lerobot_openvla/`）**：忠实复刻 OpenVLA `oxe/transforms.py` 的逐帧变换 —— 直接信任原始 action，切片取 6 个运动 DOF，**旋转保持原始表示**（四元数→欧拉角，不是轴角），**绝对位姿保持绝对**（不转 delta），**不做单位归一化**，只对 **gripper** 做规整（invert / clip，使 +1=张开 / 0=闭合）。不过滤退化帧。

> 关键差异速记：旋转 **轴角 vs 欧拉角**；绝对位姿 **转 delta（ours） vs 保持绝对（openvla：ucsd_kitchen、iamlab）**；单位 **归一化到 SI（ours） vs 原样（openvla：ucsd_kitchen 留 mm/deg、taco_play 留 CALVIN 归一化命令）**；动作来源 **state 派生（ours，多数） vs 原始 action（openvla）**。

---

## 表 1 · 我们的格式（state 派生 7D 物理 delta，轴角，SI 单位）

| 数据集 | 原始 action 格式 | 使用的转换 | 转换后动作（7D） |
|---|---|---|---|
| **viola** | dict：`world_vector(3)`+`rotation_delta(3)`+`gripper_closedness_action(1)`，是 OSC 归一化命令(±1, ~100× 物理) | 用 `ee_states`(列主序 4×4 齐次矩阵) 的相邻 EE 位姿差；旋转走四元数相对→轴角；gripper 取原始命令 | `[Δxyz(m), 轴角(rad), gripper]` |
| **stanford_hydra** | `action 7D [Δxyz, Δrpy, grip]`(OSC 命令)；`state 27D[pos,quat_xyzw,euler,...]` | state ΔEE（`state[0:3]` 位置 + `state[3:7]` 四元数，按 xyzw→wxyz 重排）；轴角；gripper=raw `action[6]` | `[Δxyz(m), 轴角, gripper]` |
| **austin_buds** | `action 7D`；`state 24D[7 joint,1 grip,16 列主序齐次]` | state ΔEE（`state[8:24]` 齐次矩阵）；轴角；gripper=raw `action[6]` | `[Δxyz(m), 轴角, gripper]` |
| **austin_sailor** | `action 7D`；`obs.state_ee 16D` 列主序齐次 | state_ee ΔEE；轴角；gripper=raw `action[6]` | `[Δxyz(m), 轴角, gripper]` |
| **austin_sirius** | `action 7D`；`obs.state_ee 16D`（少数帧全 0） | state_ee ΔEE；轴角；gripper=raw `action[6]`；**退化帧丢弃** | `[Δxyz(m), 轴角, gripper]` |
| **utaustin_mutex** | `action 7D`；`state 24D[7 joint,1 grip,16 齐次]` | state ΔEE（`state[8:24]`）；轴角；gripper=raw `action[6]` | `[Δxyz(m), 轴角, gripper]` |
| **furniture_bench** | `action 8D[Δxyz, quat(4), grip]`(quat 为速度类，几何上有歧义)；`state 35D` | state ΔEE（`state[0:3]` 位置 + `state[3:7]` 四元数 wxyz）；轴角；gripper=`state[34]` 开合宽度 | `[Δxyz(m), 轴角, gripper]` |
| **iamlab_cmu_pickup_insert** | `action 8D **绝对** [xyz, quat_wxyz, grip]` | **绝对→delta**（look-ahead，四元数相对→轴角）；gripper=raw `action[7]` | `[Δxyz(m), 轴角, gripper]` |
| **taco_play** | dict：`rel_actions_world 7D[Δxyz,Δrpy,grip]`(CALVIN 归一化, ~40–50× 物理)；`state robot_obs 15D` | state ΔEE（`robot_obs[0:3]` 位置 m + `[3:6]` 欧拉→四元数）；轴角；gripper=`rel_actions_world[6]`(±1) | `[Δxyz(m), 轴角, gripper]` |
| **nyu_franka_play** | `action 15D[7 joint_vel, Δxyz, Δrpy, grip, terminate]`；`state 13D[7 joint, xyz, rpy]` | state ΔEE（`state[7:10]` 位置 + `state[10:13]` 欧拉→四元数；**不用**命令的 Δrpy，因其按分量差非真实相对旋转）；轴角；gripper=raw `action[13]` | `[Δxyz(m), 轴角, gripper]` |
| **ucsd_kitchen** | `action 8D **绝对** [xyz(mm), euler(deg), grip, terminate]`；`state 21D` 关节 | **绝对→delta** + **mm→m, deg→rad**；四元数相对→轴角；gripper 原样 | `[Δxyz(m), 轴角(rad), gripper]` |
| **ucsd_pick_and_place** | `action 4D[3 lin_vel(归一化±1), grip torque]`；`state 7D[pos,euler,finger]` | state ΔEE（`state[0:3]` 位置 + `[3:6]` 欧拉→四元数）；轴角；gripper=raw 力矩命令 | `[Δxyz(m), 轴角, gripper]` |
| **cmu_franka_exploration** | `action 8D[Δxyz, Δrpy, grip, terminate]`（已是 delta）；无 state | 去掉 terminate；把欧拉 Δrpy 转**轴角**；gripper 原样 | `[Δxyz(m), 轴角, gripper]` |

共同点：输出全部为 **`[Δxyz(m), 轴角旋转(rad), gripper]`**；单位统一 SI；旋转用四元数相对运算 + 轴角表示。

> **gripper 说明**：上表「使用的转换」里写的 gripper 来源是历史值；现在 gripper 通道统一覆盖为 **`[0,1]`（+1=张开 / 0=闭合）**，采用与 OpenVLA 相同的每数据集规整逻辑（见下文表 2 的 gripper 处理）。两点不同：① `furniture_bench` 改用 action 命令位（而非 state 开合宽度）以落入 `[0,1]`；② `ucsd_pick_and_place` 的力矩 `[-1,1]` 经实测「正力矩=张开」，按 `(t+1)/2` 映射到 `[0,1]`（这是唯一与 OpenVLA 格式不同的 gripper —— OpenVLA 把它原样留在 `[-1,1]`）。

---

## 表 2 · OpenVLA 格式（逐帧复刻 OpenVLA 原始 action 变换，欧拉角，原始单位）

| 数据集 | 原始 action 格式 | 使用的转换（OpenVLA `oxe/transforms.py`） | 转换后动作（7D） |
|---|---|---|---|
| **viola** | dict：`world_vector(3)`+`rotation_delta(3)`+`gripper(1)` | `[world_vector, rotation_delta, invert(clip(grip,0,1))]` | `[Δxyz, Δrpy(欧拉), gripper]`（归一化命令单位，未做单位换算） |
| **stanford_hydra** | `action 7D[Δxyz,Δrpy,grip]` | `[action[:6], invert(action[6])]` | `[Δxyz, Δrpy(欧拉), gripper]`（OSC 命令单位原样） |
| **austin_buds** | `action 7D` | `[action[:6], invert(clip(action[6],0,1))]` | `[Δxyz, Δrpy(欧拉), gripper]` |
| **austin_sailor** | `action 7D` | `[action[:6], invert(clip(action[6],0,1))]` | `[Δxyz, Δrpy(欧拉), gripper]` |
| **austin_sirius** | `action 7D`（少数帧 state 全 0） | `[action[:6], invert(clip(action[6],0,1))]`；**不丢弃退化帧** | `[Δxyz, Δrpy(欧拉), gripper]` |
| **utaustin_mutex** | `action 7D` | `[action[:6], invert(clip(action[6],0,1))]` | `[Δxyz, Δrpy(欧拉), gripper]` |
| **furniture_bench** | `action 8D[Δxyz, quat(4), grip]` | `[action[:3], quat_xyzw→欧拉(action[3:7]), invert(clip(grip,0,1))]`（把 quat 当 xyzw，直接转欧拉） | `[Δxyz, 欧拉, gripper]` |
| **iamlab_cmu_pickup_insert** | `action 8D **绝对**[xyz, quat, grip]` | `[action[:3], quat_xyzw→欧拉, action[7]]`；**保持绝对、不转 delta**，gripper 原样 | `[xyz(绝对), 欧拉(绝对), gripper]` |
| **taco_play** | dict：`rel_actions_world 7D` | `[rel[:6], clip(rel[6],0,1)]`（不 invert） | `[Δxyz, Δrpy(欧拉), gripper]`（CALVIN 归一化命令单位原样） |
| **nyu_franka_play** | `action 15D` | `[action[-8:-2] (Δxyz,Δrpy), clip(action[-2:-1] grip,0,1)]` | `[Δxyz, Δrpy(欧拉), gripper]` |
| **ucsd_kitchen** | `action 8D **绝对**[xyz(mm), euler(deg), grip, terminate]` | `action[:-1]`（去 terminate）；**保持绝对、mm/deg 原样、不转 delta** | `[xyz(mm,绝对), euler(deg,绝对), gripper]` |
| **ucsd_pick_and_place** | `action 4D[3 lin_vel, grip]` | `[action[:3], zeros(3), action[3]]`（旋转补 0） | `[lin_vel(3), 0,0,0, gripper]` |
| **cmu_franka_exploration** | `action 8D[Δxyz, Δrpy, grip, terminate]` | `action[:-1]`（去 terminate） | `[Δxyz, Δrpy(欧拉), gripper]` |

夹爪约定：OpenVLA 统一把 gripper 规整成 **+1=张开 / 0=闭合**；`invert`=`1−x`，`clip`=截断到 `[0,1]`。
四元数：OpenVLA 用 `tensorflow_graphics.euler.from_quaternion`（标量在后 `xyzw`，返回 XYZ 欧拉）；本仓库已用等价 NumPy 实现复刻。

---

## 最容易踩坑的几处差异（实验时重点看）

| 数据集 | ours | openvla |
|---|---|---|
| **iamlab** | 绝对位姿 → **delta**，轴角 | **保持绝对位姿**，欧拉 |
| **ucsd_kitchen** | 绝对 → delta，且 **mm→m / deg→rad** | **保持绝对**，**mm/deg 原样** |
| **taco_play** | state 派生物理 delta（m/rad） | rel 命令原样（**CALVIN 归一化单位**，量级差 ~40–50×） |
| **nyu_franka_play** | state 派生旋转（命令 Δrpy 不可靠，弃用） | 直接用命令的 Δrpy |
| **furniture_bench** | 用 state 真实四元数算 delta | 把 action 里的 quat 速度场直接当 xyzw 转欧拉 |
| **austin_sirius / buds** | 退化帧（全 0 齐次矩阵）**丢弃** | **保留**全部帧 |
| 旋转表示 | 轴角 rotvec | 欧拉角（部分为绝对欧拉） |
