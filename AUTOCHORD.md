# AutoChord / Chordino 引擎说明

> 实验性特性：依赖体积大，且部分组件需要手动安装/编译（尤其是 vamp 模块和 Chordino 插件）。Windows 建议使用独立虚拟环境。

## AutoChord（神经网络）
- 安装：
  ```bash
  pip install scipy librosa vamp lazycats gdown tensorflow-cpu soundfile
  pip install git+https://github.com/cjbayron/autochord
  ```
- 运行：`python main.py --engine autochord`
  - 首次运行会通过 gdown 下载模型（需联网）。
  - 推理失败会在控制台提示，可随时切回 `--engine simple`。
- 限制：上游只提供 Linux NNLS-Chroma VAMP 插件；Windows 如无可用插件，AutoChord 可能无法运行。

## Chordino（VAMP）
- 安装依赖：
  ```bash
  pip install vamp librosa soundfile
  ```
- 安装 Chordino 插件：使用官方 Vamp Plugin Pack（Windows 版），确保其中的 chordino/nnls-chroma 插件可用，并记下插件目录。
- 运行：
  ```bash
  python main.py --engine chordino --vamp-path <VAMP插件目录>
  ```
  或预先设置环境变量 `VAMP_PATH=<VAMP插件目录>`。
- Windows 需要 MSVC Build Tools 编译 `vamp` 模块；如安装失败，可在 WSL/Linux 使用 chordino 引擎，或退回 `--engine simple`。
