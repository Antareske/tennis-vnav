# tennis-vnav — 单目视觉网球导航与数采系统

基于 SG2002 小车的单目视觉网球导航系统，为 ACT（Action Chunking Transformer）模型训练提供数据采集能力。导航逻辑为 YOLO 检测 + 状态机；数据以 LeRobot 兼容格式落盘，图像经板载 VENC 硬件编码，采样频率 10 FPS。

## 硬件平台

| 组件 | 型号 | 接口 |
|------|------|------|
| 主控 | SG2002 (cv181x, RISC-V 64, 单核 C906) | — |
| 摄像头 | USB UVC 1e45:8022（YUYV 原生，~9.4 FPS 上限） | 640×480 |
| 电机 | ESP32-C3 + TT 马达 | /dev/ttyS1, 115200 |
| 推理 | CVI TPU 0.5 TOPS | /dev/cvi-tpu0 |
| 编码 | VENC 硬件 JPEG | /dev/cvi_vc_enc0 |

## 系统架构

```
手机浏览器 ──HTTP(80)── ctrl-serve（Rust，HTML 内嵌，~2MB 内存）
                            │ /tmp/vnav/ 文件通信
                            ▼
导航进程 main.py ──► 摄像头 YUYV ──┬─► cv2 转 BGR ─► YOLO ─► 状态机 ─► 电机
                                    └─► YUYV→NV12 字节重排 ─► VENC 硬件编码 ─► JPEG 落盘
```

- **数采线程**：独立线程时间驱动（10 Hz），与导航主循环解耦；图像先写 tmpfs 暂存，episode 结束时搬移至 SD
- **导航主循环**：YOLO tick 限频 ~6.7 Hz（150ms），为单核系统让出数采预算
- **控制协议**：`/tmp/vnav/cmd.txt`（start/abort/clear）+ `status.json`，ctrl-serve 与导航进程通过文件通信，互不阻塞

## 当前实现情况

| 模块 | 说明 |
|------|------|
| `main.py` | 多轮采集主循环：IDLE 等待命令 → 采集 → 保存 → 回 IDLE |
| `vnav_control.py` | 与 ctrl-serve 的文件通信（命令消费、状态原子写） |
| `state_machine.py` | SEARCH → OBSERVE → APPROACH → DONE；含横向死区与 PWM 自适应（掉电/打滑补偿） |
| `data_collector.py` | 异步 10 FPS 采样线程，tmpfs 暂存，meta.json 含抖动统计 |
| `hwjpeg_enc.py` / `hwjpeg.c` / `libhwjpeg.so` | VENC 硬件 JPEG 编码封装（ctypes） |
| `ctrl-serve/` | Rust 网页控制服务（无依赖，HTML 编译进二进制） |
| `motor_tt_pid.py` | ESP32 UART 协议（串口锁串行化 + RPM 缓存） |

### 性能指标（板端实测）

| 指标 | 值 |
|------|-----|
| 数采采样率 | 10.0 FPS（均值间隔 99.9ms，std ~30ms） |
| 硬件编码耗时 | ~16ms/帧（不含 YUYV→NV12 重排 ~5ms） |
| 导航主循环 | ~6.7 Hz tick（YOLO 推理 84ms） |
| 图像 | 640×480 JPEG，质量 60，~65-85KB/帧 |

### 数据格式（LeRobot 兼容）

```
data/episode_NNN/
├── images/frame_XXXXXX.jpg   # JPEG 图像
├── states.npy                # [N,2] float32 实际轮速 RPM
├── actions.npy               # [N,2] float32 PWM 指令
├── timestamps.npy            # [N] float64 采样时刻
└── meta.json                 # 元信息 + 采样抖动统计
```

Episode 结构：静止 0.5s → 导航全程 → 静止 1s，首尾帧 action=(0,0)。

## 所需资产

### 开发机

| 资产 | 用途 |
|------|------|
| `/opt/riscv64-linux-musl-cross` | RISC-V musl 交叉工具链（编译 hwjpeg.so 与 ctrl-serve） |
| nightly Rust 工具链 | ctrl-serve 交叉编译（`-Zbuild-std` 模式） |
| `hwjpeg-sdk/`（仓库内） | sophgo cvi_mpi（sg200x-dev 分支）中间件头文件，hwjpeg.c 编译依赖 |

### 板端

板端全部运行资产随仓库分发，部署脚本统一上传，**不依赖 AKA-00 等外部项目**：

| 资产 | 仓库路径 | 板端路径 |
|------|---------|---------|
| CVI 运行时 | `cvi-libs/libcviruntime.so`、`libcvikernel.so` | `/root/tennis-vnav/cvi-libs/` |
| 中间件 | `cvi-libs/libsys.so`、`libvenc.so` | `/root/tennis-vnav/cvi-libs/` |
| 原子库 | `cvi-libs/libatomic.so.1` | `/root/tennis-vnav/cvi-libs/` |
| VENC 封装 | `libhwjpeg.so` | `/root/tennis-vnav/` |
| 控制服务 | `ctrl-serve/ctrl-serve` | `/root/tennis-vnav/` |
| YOLO 模型 | `models/tennis.onnx`、`models/yolov8n_tennis_v2.cvimodel` | `/root/tennis-vnav/models/` |

纯净镜像自带（部署前置条件，README 不负责安装）：

| 资产 | 说明 |
|------|------|
| Python 3.11 + OpenCV + numpy | 板端图像处理与推理预处理 |
| pyserial | ESP32 串口通信 |
| BusyBox + hostapd + udhcpd | init 系统与 AP 热点（S98apstart） |

运行时库加载顺序：仓库 `cvi-libs/` 优先，回退镜像系统路径（`/usr/bin/lib`、`/usr/bin/dl_lib`）。

## 构建流程

```bash
# 1. hwjpeg.so（VENC 硬件编码封装）
cd tennis-vnav
/opt/riscv64-linux-musl-cross/bin/riscv64-linux-musl-gcc \
  -D__CV181X__ -I hwjpeg-sdk -O2 -shared -fPIC -o libhwjpeg.so hwjpeg.c

# 2. ctrl-serve（网页控制服务）
cd ctrl-serve
cargo build --release --target riscv64gc-unknown-linux-musl -Zbuild-std=std,panic_abort
# 产物: target/riscv64gc-unknown-linux-musl/release/ctrl-serve
```

## 部署流程

纯净镜像 + 仓库即可部署（无需 AKA-00 等外部项目）：

```bash
# 0. 板端一次性初始化（新镜像首次）：安装 AP 热点 + 禁用 AKA-00 自启（幂等）
./scripts/board-setup.sh                     # 默认 root@192.168.4.1

# 1. 一键部署（上传全部资产 + 安装自启 + 重启服务）
./scripts/deploy.sh                          # 默认 root@192.168.4.1
./scripts/deploy.sh --no-restart             # 仅上传，重启板子后生效
```

部署完成后：重启板子（或脚本已重启服务），开机流程为
`S98apstart（AP 热点）→ S99vnav（UART pinmux + ctrl-serve + main.py 双保活）`。
AKA-00 不再开机运行（其自启脚本被移出 init.d，备份于 `/root/S99webstart.disabled`）。

手动启动（调试用）：

```bash
ssh root@192.168.4.1
devmem 0x03001064 32 0x6 && devmem 0x03001068 32 0x6   # UART pinmux
/root/tennis-vnav/ctrl-serve &
cd /root/tennis-vnav && python3 main.py                # 库路径由代码自动解析
```

## 使用与测试流程

1. 小车通电，等待 ~30s（AP 热点就绪）
2. 手机连接热点，浏览器访问 `http://192.168.4.1`
3. 摆好网球，点击「开始采集」→ 小车导航并采集
4. 导航完成（或点击「中止」）后数据保存，页面回到空闲
5. 效果不满意时点击「删除第 N 组」丢弃最近一组
6. 数据回传 PC：

```bash
ssh root@192.168.4.1 "cd /root/tennis-vnav/data && tar cf /tmp/ep.tar episode_NNN"
ssh root@192.168.4.1 "cat /tmp/ep.tar" > ep.tar && tar xf ep.tar
```

7. 质量检查：`meta.json` 中 `effective_fps ≥ 9.5`、`sampling_jitter.std_interval_ms ≤ 40` 为合格

## 关键实现要点

- **VENC 输入交接**：`VIDEO_FRAME_INFO_S` 必须显式设置全部平面地址（plane 0/1），只填 plane 0 会导致硬件从无效地址读色度、整帧花屏
- **VENC 不接受 RGB 输入**：需转换为 NV12（或 YUV420）；摄像头原生 YUYV 到 NV12 为纯字节重排，无色彩运算
- **VB 缓冲为 uncached 映射**：逐像素写入极慢（~300ms），需先在缓存内存转换再 memcpy
- **中间件初始化顺序**：`VB_SetConfig → VB_Init → SYS_Init`（顺序颠倒会返回 VB_ILLEGAL_PARAM）
- **板端 SDK 差异**：VB 配置要求 `u32MaxPoolCnt ≥ 1`（官方 sample 的 0 池配置在板端 SDK 返回非法参数）
- **摄像头帧率锁 10 FPS**：vendor UVC 驱动未实现 S_PARM（设帧率 ioctl），硬件虽支持 30 FPS 但无法从用户态切换，数采 10 FPS 已与图像源匹配
- **串口并发**：多线程共用串口需加锁，否则 `reset_input_buffer` 会清掉其他线程等待的响应帧
