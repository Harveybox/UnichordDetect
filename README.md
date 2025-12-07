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

## 安装
1. 安装 Python 3.10+，并确保系统为 Windows 且启用 WASAPI Loopback。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

> 若未安装 PySide6，可单独 `pip install PySide6`。

## 运行
- 列出可用 loopback 设备：
  ```bash
  python main.py --list-devices
  ```
- 直接使用默认设备：
  ```bash
  python main.py
  ```
- 指定设备索引：
  ```bash
  python main.py --device 5
  ```

运行后将出现置顶半透明窗，显示当前和弦与最近时间轴，可拖拽位置，点击 `Close` 退出。Ctrl+C 也会优雅退出采集与识别线程。

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
