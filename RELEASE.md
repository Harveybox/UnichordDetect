# UnichordDetect 发布包说明

## 版本
- 当前版本：`0.1.0`
- 版本号存于根目录 `VERSION`；更新版本时请同步修改。

## 如何获取/生成 Release 压缩包
- **从 GitHub 下载**：在仓库页面使用 **Code → Download ZIP**，或从 Release 附件直接下载 `UnichordDetect-<version>.zip`。
- **本地生成**：在 Windows 或任意支持 Python 3.10+ 的环境下执行：
```bash
python scripts/make_release.py
```
脚本会在本地 `dist/` 目录生成 `UnichordDetect-<version>.zip`，包含：
- 全部源码：`audio_capture/`, `chord_estimator/`, `timeline/`, `ui/`, `main.py`
- 说明：`README.md`, `RELEASE.md`, `VERSION`, `requirements.txt`

> 仓库不收录任何压缩包/二进制；若要在 CI 构建，请运行脚本并上传 `dist/UnichordDetect-<version>.zip` 作为流水线产物或 Release 附件。

> 默认以源码/zip 分发，暂未内置 `.exe`。如需单文件可执行，请在 Windows 上安装依赖后使用 `pyinstaller --noconsole --onefile main.py` 自行打包。

## 下载/解压与运行
1. 解压 zip 至任意路径，例如 `C:\UnichordDetect`。
2. 在解压目录打开终端，安装依赖：
   ```bash
   python -m pip install -r requirements.txt
   ```
3. 列出 loopback 设备（可选）：
   ```bash
   python main.py --list-devices
   ```
4. 启动应用（默认设备或指定索引）：
   ```bash
   python main.py
   # 或
   python main.py --device <index>
   ```

## 功能测试 checklist
- **音频采集**：运行 `--list-devices` 有输出；启动后无异常抛出，命令行未报 `No WASAPI loopback devices found`。
- **实时和弦**：播放含清晰和弦的音源（如 C/F/G 大三和弦），观察右上角标签在 0.3–0.5 秒内稳定显示；静音时显示 `N`。
- **平滑/抖动抑制**：在和弦切换时标签不应快速闪烁，可通过调整 `DEFAULTS` 中 `smoothing_frames`/`min_confirm_seconds` 验证效果。
- **时间轴滚动**：彩色条带从右向左滚动，最近 60–120 秒内的和弦均可见；拖动窗口应无标题栏、始终置顶。
- **关闭/退出**：点击 `Close` 或 `Ctrl+C` 后，窗口关闭且进程退出，无残留 Python 进程。

## 常见故障与诊断
- **未找到 loopback**：确认在 Windows 本机（非 WSL）运行，声卡属性中启用“立体声混音/回放”；必要时更新声卡驱动。
- **依赖缺失**：出现 `pyaudiowpatch is required for WASAPI loopback` 或 Qt 模块错误时，重新执行 `pip install -r requirements.txt`。
- **UI 不显示或黑屏**：检查是否在远程桌面/无 GPU 环境；尝试更新显卡驱动或在本地桌面运行。
- **高延迟/卡顿**：减小 `DEFAULTS` 的 `window_seconds` 或 `ui_display_seconds`，关闭其他高占用程序。
