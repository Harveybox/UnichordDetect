# UnichordDetect

通用和弦识别伴随应用（MVP）示例，使用 Windows WASAPI loopback 采集系统输出音频，实时进行和弦识别，并在桌面叠加层展示最近时间轴。

## 系统架构（文字描述）
- **音频采集**：`audio_capture/` 内的 `LoopbackCapture` 基于 pyaudiowpatch 打开 WASAPI loopback 设备，将 PCM 数据写入线程安全的环形缓冲区。
- **特征与识别**：`chord_estimator/` 中的 `ChordEstimator` 线程定期从环形缓冲区读取窗口，计算 STFT-Chroma，使用 12 个大/小三和弦模板匹配，并做多数投票与最短持续时间平滑。
- **时间轴维护**：`timeline/Timeline` 接收每次估计，按时间戳维护 `Segment(start, end, label, confidence)` 列表，并裁剪到滚动窗口。
- **UI 叠加层**：`ui/overlay_ui.py` 用 PySide6 创建置顶透明窗，定时从 `Timeline` 拉取最近 60–120 秒的分段，并绘制横向条带和当前和弦。
- **主流程**：`main.py` 启动采集与识别线程，提供设备枚举/选择，启动 Qt UI；收到退出信号后依次停止各线程。

## 模块接口概要
- `LoopbackCapture(list_loopback_devices, start, read, stop)`：枚举、打开 loopback 设备，提供 `read(num_frames)` 取出指定帧数 PCM。
- `ChordEstimator(start, stop)`：后台线程读取 PCM，`on_estimate` 回调输出 `ChordEstimate(label, confidence, timestamp)`。
- `Timeline.update(timestamp, label, confidence)`：若标签变化则关闭上一段并开始新段，`get_recent()` 复制最近窗口。
- `run_overlay_app(fetch_segments, refresh_ms, display_seconds)`：启动 Qt 主循环，每个刷新周期调用 `fetch_segments()` 更新显示。

## 默认参数（MVP）
- 采样率：48 kHz（自动适配设备默认，44100 亦可）
- 分析窗口：1.5 s；Hop：0.1 s（10 FPS）
- 平滑：5 帧多数投票 + 0.4 s 最短持续时间阈值
- UI 显示窗口：90 s（时间轴保留 120 s 便于裁剪）

## 获取代码（GitHub 下载方式）
- 方式 A：克隆仓库（保持更新方便）
  ```bash
  git clone <repo_url> UnichordDetect
  cd UnichordDetect
  ```
- 方式 B：GitHub “Download ZIP”/Release 压缩包
  - 在 GitHub 页面点击 **Code → Download ZIP**，或从 Release 附件下载 `UnichordDetect-<version>.zip`。
  - 将压缩包解压到本地目录（如 `C:\UnichordDetect`），然后在该目录打开终端。

> 本项目默认以 Python 源码方式运行；当前未提供预编译 `.exe`。如需单文件分发，可在 Windows 上使用 `pyinstaller --noconsole --onefile main.py` 自行打包（需已安装依赖和 VC++ 运行库）。

## 安装

1. 安装 Python 3.10+（建议 64-bit）。
2. Windows 平台额外建议安装 Microsoft Visual C++ 2015–2022 可再发行组件（x64），以满足 Qt 运行库依赖。
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

若未安装 PySide6，可单独 `pip install PySide6`。

## 运行
- 列出可用 loopback 设备：
  ```bash
  python main.py --list-devices
  ```
- 直接使用默认设备：
  ```bash
  python main.py
  ```
- 使用神经网络 AutoChord 引擎（需先安装可选依赖，见文末）：
  ```bash
  python main.py --engine autochord
  ```
- 使用 VAMP Chordino 引擎（需安装 `vamp` 模块和 chordino 插件，见文末）：
  ```bash
  python main.py --engine chordino --vamp-path <VAMP插件目录>
  ```
- 指定设备索引：
  ```bash
  python main.py --device 5
  ```

运行后将出现置顶半透明窗，显示当前和弦与最近时间轴，可拖拽位置，点击 `Close` 退出。Ctrl+C 也会优雅退出采集与识别线程。

## Release 打包与下载
- 版本号存放于 `VERSION`；执行 `python scripts/make_release.py` 会在本地 `dist/` 生成 `UnichordDetect-<version>.zip`，包含源码、依赖清单与说明文件，可直接分发/下载。
- 仓库不收录任何压缩包/二进制产物（`dist/` 已被忽略）；如需分发请自行运行脚本生成并上传 CI 产物或 Release 附件。打包结果中包含 `README.md` 与 `RELEASE.md` 方便终端用户安装运行。

### 在 GitHub Actions 上自动构建 Windows 可执行文件

本仓库包含一个用于在 Windows runner 上使用 PyInstaller 打包的工作流：`.github/workflows/build_windows.yml`。

- 触发方式：在仓库的 Actions -> Build Windows executable -> Run workflow（workflow_dispatch）。
- 产物：构建成功后会把 `dist/` 目录上传为工作流 artifact（名称 `UnichordDetect-windows`），可在 workflow 运行完成后下载可执行文件。

注意：首次构建可能需要针对 PySide6/依赖做调整；若在 CI 中出现缺失的动态库或插件，请把构建日志贴给我，我会帮助调整 `pyinstaller` 参数（例如 `--add-data` 指定 Qt platform plugins 的路径）。

## 手工测试与诊断流程
以下步骤可覆盖音频采集、识别平滑、时间轴与 UI 全链路，并帮助定位常见问题：

1) **环境确认**：
   - `python --version`，确认 3.10+。
   - `pip show pyaudiowpatch PySide6 numpy`，确保依赖安装；若缺失请重新安装。

2) **设备枚举**：
   ```bash
   python main.py --list-devices
   ```
   - 预期看到含有 `isLoopbackDevice=True` 的输出；若为空，请检查 Windows 声卡是否支持/启用 loopback 或声卡驱动设置。

3) **基本运行**：
   ```bash
   python main.py            # 默认 loopback
   # 或
   python main.py --device <index>
   ```
   - 观察 UI 顶部“当前和弦”实时跳变，时间轴右侧为最新片段，旧片段向左滚动。
   - 如果 UI 不出现，请查看命令行是否有 Qt 加载错误；确认显卡驱动/远程桌面环境允许 OpenGL/Qt。

4) **识别与平滑验证**：
   - 播放含清晰和弦的音频（如钢琴 C/F/G 和弦），预期标签在 0.3–0.5 s 内稳定显示对应 maj/min；无音乐或噪声时应显示 `N`。
   - 若标签频繁抖动，可调整 `DEFAULTS` 中 `smoothing_frames` 或 `min_confirm_seconds`，或检查输入信噪比。

5) **时间轴检查**：
   - 拖动或拖拽窗口位置，确保无边框置顶效果；查看时间轴色块与文字是否连续、无断裂；在 60–120 s 范围内自动滚动。

6) **诊断日志与故障排查**：
   - 运行时若抛出 `No WASAPI loopback devices found`，请确认在 Windows 本机运行而非 WSL，并检查声卡设置。
    - 若提示 `pyaudiowpatch is required for WASAPI loopback`，请确保通过 `pip install pyaudiowpatch` 安装并在 Windows 环境运行。
    - 若启动时报 `ImportError: DLL load failed while importing QtCore`，通常是 32/64 位架构不匹配或缺少 VC++ 运行库：
       - 使用 64-bit Python 并重新安装 PySide6：`pip install --force-reinstall PySide6==6.7.3`。
       - 确认已安装 Microsoft Visual C++ 2015–2022 x64 运行库，若缺失请从微软官网下载安装。
   - 若 UI 卡顿，可降低 `DEFAULTS` 中 `ui_display_seconds` 或 `window_seconds`，并关闭其他占用 GPU 的窗口。

## 常见问题
- **找不到 loopback 设备**：确保在支持 WASAPI 的 Windows 上运行，并使用 pyaudiowpatch；某些设备需要在声卡属性中启用“立体声混音/回放”。
- **采样率不匹配**：默认使用设备的 `defaultSampleRate`；如设备为 44.1k，会自动按设备速率运行算法。
- **无声输入**：算法输出 `N`（无和弦）；UI 会显示灰色条段。
- **性能与延迟**：端到端目标 <300 ms，采用 100 ms hop 和 1.5 s 窗口；UI 刷新 30 fps。可根据需要在 `DEFAULTS` 中调整窗口、hop、平滑参数。

## 目录结构
- `audio_capture/`：WASAPI loopback 采集与环形缓冲
- `chord_estimator/`：Chroma 计算 + 模板匹配
- `timeline/`：Segment 时间轴管理
- `ui/`：PySide6 叠加层
- `main.py`：应用入口
- `requirements.txt`：依赖列表

## 后续扩展接口
- 在 `ChordEstimator` 可替换特征与模型（Essentia/madmom 等）。
- `Timeline` 可加入“回填修正”逻辑（Viterbi/重算最近 1–2 s）。
- 可在采集层添加按应用抓取或导出 JSON/SRT/CSV 的接口。
