# Meet Notes：本地会议转录与 AI 纪要

Windows / PySide6 桌面程序：分别录制系统声音和麦克风，在本机转录中日英、区分说话人并截图。可选用 DeepSeek 生成 Markdown 纪要，再同步到 Notion。**录音不上传**；DeepSeek 接收转录文字，Notion 接收纪要。

## 1. 配置环境（首次使用）

1. 安装 **Python 3.11 64 位**，保留 Python Launcher（`py`）；执行 `py -3.11 --version` 确认。下载或克隆项目，解压到可写目录。
2. 双击 **`setup.bat`**，等待安装成功。脚本创建 `.venv`、安装基础依赖，并从 `.env.example` 创建 `.env`，不会覆盖已有配置。
3. 按下表选择附加安装。安装、更新前先退出程序，完成后重启。

| 推理方式 | 在 `setup.bat` 成功后运行 | 程序内选择 |
| --- | --- | --- |
| Whisper / SenseVoice CPU | 无需额外运行库 | 推理设备 `CPU` |
| Whisper NVIDIA GPU | `setup-gpu.bat` | `Whisper` → `NVIDIA GPU` → `float16` |
| SenseVoice NVIDIA GPU | `setup-sensevoice-gpu.bat`（包含 GPU 运行库） | `SenseVoiceSmall` → `Small · FP32` → `NVIDIA GPU` |

GPU 需要 NVIDIA 显卡和可用驱动。脚本将 **CUDA 12 / cuDNN 9** 等运行库装入 `.venv`，不修改系统 CUDA 或系统 PATH；仅装系统 CUDA 13 不能代替它们。无需安装 PyTorch。

4. 双击 **`run.bat`**，准备模型：Whisper 首次开始会议时自动下载；SenseVoice 在「录制 → 转录模型」点下载按钮；说话人模型在「说话人」点下载按钮，也可运行 `setup-speakers.bat`。
5. 选好音频设备，使用「硬件测试与回放」检查麦克风和系统声音。Windows 须允许桌面应用访问麦克风。

首次下载依赖、模型需要联网；准备完成后录音、转录和说话人分析可离线运行。DeepSeek、Notion 可留空。所有脚本固定使用 **`.venv\Scripts\python.exe`**，无需手动激活虚拟环境。

## 2. 启动会议、结束与退出

1. 双击 **`run.bat`**，确认设备、模型、语言及实际发言人数。默认开启两路录音和说话人分析，总人数为 3，请按本次会议调整。
2. 点 **「开始会议」**，等状态变为 **「正在录音」** 再发言。模型下载、加载和 GPU 预检期间尚未录音。
3. 会议中按 **F9 /「重点截图」** 保存画面；右侧音量条和待转录时长显示采集、处理进度。
4. 结束时按 **F8 /「结束并补完转录」**，等待剩余转录和说话人分析完成。不想等待可点 **「仅保存录音」**，以后打开历史会议再「重新转录」。
5. 按需点 **「生成纪要」**，再在 Notion 分类点 **「同步当前会议纪要」**；也可提前配置自动执行。
6. 确认任务结束后关闭窗口。直接关闭正在处理的窗口会停止后台任务并保留已有数据，未完成的转录需稍后恢复。

也可在项目目录的 PowerShell 执行 `.\.venv\Scripts\python.exe main.py`。`run.bat` 在虚拟环境缺失时会先调用安装脚本。

## 3. 软件设置：设计逻辑与配置方法

左侧分类可以展开、折叠，折叠不会清空内容。**DeepSeek / Notion 点击保存后写入 `.env`；其他录制参数、自动生成纪要勾选状态和折叠状态目前不跨次启动保存。** 设备、模型和人数应在开始会议前设置。

### 录制 → 音频设备

| 设置 | 为什么这样设计 / 怎么配置 |
| --- | --- |
| 会议播放声 / 扬声器 | 环回录制所选输出设备的全部声音。让会议软件输出到同一设备；会包含其他应用声音，截图窗口的选择不限制录音范围。 |
| 自己的麦克风 / 麦克风 | 保存独立音轨，按需勾选并选设备。建议戴耳机，避免外放声音再次录入；程序没有回声消除。 |
| 刷新设备与窗口 | 插拔设备、打开会议窗口后刷新并重新选择。至少启用一路录音。 |

### 录制 → 转录模型

| 设置 | 为什么这样设计 / 怎么配置 |
| --- | --- |
| 识别网络 / Whisper 模型 | Whisper 支持 CPU / GPU，提供 `tiny`、`base`、`small`、`medium`、`large-v3`，也可输入兼容模型名或本地路径。默认 `small`；大模型通常需要更多内存和计算时间。 |
| SenseVoice 版本 | `Small · INT8` 为 CPU 量化版（约 239MB），`Small · FP32` 为 CPU / GPU 浮点版（约 938MB）。选好版本后点击「下载 / 检查所选 SenseVoice 模型」。 |
| 主要语言 | 中日英切换选「中日英混说（逐段检测）」，按语音停顿分段判断语言并保留原文；单语会议可固定中文、日语或英语。「自动检测」使用引擎常规检测。 |
| 推理设备 / 计算精度 | Whisper CPU 默认 `int8`，GPU 默认 `float16`；也可选 `int8_float16`、`float32`，须设备支持。SenseVoice 精度随版本决定：选 GPU 自动切 FP32，选 INT8 自动切 CPU。 |
| CPU 线程 | 默认 4，控制转录引擎 CPU 线程；机器卡顿时可调小，不控制独立的说话人进程。 |
| 识别提示 | 人名、项目名、专业术语可辅助 Whisper；SenseVoice 不使用此项。 |

两个 SenseVoice 版本都属于 Small，FP32 没有增大参数规模。更大模型的公开情况以 [官方仓库](https://github.com/QwenAudio/SenseVoice) 为准。快速混说、重叠发言和术语仍可能误识别，重要内容需复核。

### 录制 → 截图与保存

| 设置 | 为什么这样设计 / 怎么配置 |
| --- | --- |
| 截图窗口 / 整屏截图 | 捕获整个所选窗口，可能包含标签栏；会议期间请保留目标标签页。需要桌面时显式勾选整屏截图，窗口捕获失败不会自动改录桌面。 |
| 自动截图间隔 | `0` 关闭，启用时最少 5 秒；也可按 F9 手动截图，热键被占用时用按钮。 |
| 保存目录 | 默认 `sessions/`，每场会议独立建目录；可改到容量充足的位置。双路 WAV 约 230MB/小时，另加截图。 |

### 说话人

| 设置 | 为什么这样设计 / 怎么配置 |
| --- | --- |
| 会议中同时区分说话人 | 独立 CPU 进程（2 线程）与转录同时运行，不额外占显存。首次下载说话人模型（约 46MB）；文字先出现，约 24 秒后开始补暂定归属。 |
| 总开会人数（含自己） | 填**实际发言人数**，`0` 自动估计。人数用于整场声纹聚类，随新发言修正历史归属，结束时统一分组；填错可能误拆分或合并。 |
| 自己的麦克风单独占一人 | 录麦克风且勾选时，系统音轨按总人数减一分组。例如共 3 人，自己用麦克风，系统声按 2 人聚类。仅录系统声时填该音轨人数；多人共用麦克风时不要勾选。 |
| 姓名 / 选择声音样本并登记 | 用干净的单人录音关联姓名，建议每人 30～60 秒，支持 WAV、FLAC、MP3、M4A、OGG。同名登记会替换；声纹只存在本机 `speaker-profiles.json`。 |
| 当前会议说话人 / 修改姓名 | 会后选编号、填姓名并保存，只修正当前会议。之后重新生成纪要，再同步 Notion。 |

未登记姓名时显示编号；无法提取声纹的片段标“未知”，重叠人声单独标注。姓名匹配仍需核对。说话人分析失败单独报告，不中断录音和文字转录。

### DeepSeek：转录文字 → Markdown 纪要（可选）

在 [DeepSeek 平台](https://platform.deepseek.com/) 创建 API Key，填写后点「保存 API 配置」。默认值与 [官方 API 文档](https://api-docs.deepseek.com/) 对齐，也可按账号可用模型调整。

| 界面设置 | `.env` 配置 | 用途 / 默认值 |
| --- | --- | --- |
| API Key | `DEEPSEEK_API_KEY` | 填自己的密钥；留空不能生成纪要。 |
| API 地址 | `DEEPSEEK_BASE_URL` | `https://api.deepseek.com`，须 HTTPS；文字将发到此地址。 |
| 模型名称 | `DEEPSEEK_MODEL` | `deepseek-flash`，与本地语音识别模型独立。 |
| 请求超时 | `DEEPSEEK_TIMEOUT` | 默认 120 秒，范围 5～600 秒；网络慢时可调大。 |
| 会议结束后自动生成纪要 | 无，仅当前启动期间生效 | 勾选后自动生成，否则手动点「生成纪要」。 |

长会议分段整理再合并，成功分段本地缓存以便重试，完整结果写入 `minutes.md`。失败或取消保留已有内容；不完整转录生成的纪要会注明。这会产生 API 用量，音频、截图不上传。

### Notion：纪要 → 指定页面（可选）

1. 按 [Internal connection 官方说明](https://developers.notion.com/guides/get-started/internal-connections) 创建内部连接，开启**读取、插入、更新内容**权限。
2. 在目标普通页面的 `••• → Connections → Add connection` 添加该连接，授予页面访问权。
3. 在程序填 Token、页面完整链接或 ID，保存后「测试连接」。测试只验证读取，首次上传才验证写权限。
4. 生成纪要后「同步当前会议纪要」，或勾选自动同步并保存。

| 界面设置 | `.env` 配置 | 用途 |
| --- | --- | --- |
| Token | `NOTION_API_KEY` | Internal connection Token。 |
| 目标页面 | `NOTION_PARENT_PAGE_ID` | 每场会议在此处创建子页面，不能直接填数据库链接。 |
| 生成纪要后自动同步到 Notion | `NOTION_AUTO_SYNC` | `true` 自动，默认 `false`；先完成纪要生成才触发同步。 |

重复同步不重复创建，更新纪要会更新对应子页面。远端正文被手动修改时停止覆盖，需先核对、合并。只上传 `minutes.md`（最多 350KB），本地截图链接不会自动成为 Notion 图片。

**密钥保存：** DeepSeek、Notion 密钥在 `.env` 中明文保存，输入框遮挡显示，系统环境变量优先。分享项目仅附带空白模板 `.env.example`，不要附带 `.env`。

### 硬件测试与回放

先在「音频设备」选设备，再选择测试类型；不加载模型，与正式录音互斥。

| 测试 | 操作与判断方法 |
| --- | --- |
| 麦克风 | 开始测试 → 说几句话 →「停止测试并回放」。听是否断续、变速，对照录音时长与经过时间；最长 120 秒。 |
| 系统播放声 | 自动向所选扬声器播放三个短测试音，约 5 秒后停止并回放。检查是否录到测试音及匹配度，匹配度不代表音质评分。 |
| 再次回放 / 停止回放 | 重听最近文件或停止；回放使用 Windows 默认输出设备，文件在 `test-output/hardware/`。 |

## 4. 文件、恢复与依赖核对

会议目录包含 `system.wav` / `mic.wav`（启用的音轨）、`events.jsonl`（事件）、`transcript.txt` / `transcript.json`（转录）、`screenshots/`、`minutes.md`（生成后）、`notion-sync.json`（同步后）。

「打开历史会议」会从事件日志恢复文字；「重新转录」按当前模型、语言、说话人设置处理保存的音频，将结果放进 `recovered_.../`，保留原始结果。已丢失的声音无法补回。SenseVoice 权重在 `models/sensevoice/`，说话人权重在 `models/speakers/`，Whisper 使用 Hugging Face 默认缓存；模型、密钥、会议数据、测试输出和虚拟环境不上传源码仓库。

| 文件 | 与安装脚本的对应关系 |
| --- | --- |
| `requirements.lock.txt` | `setup.bat` 优先使用的已验证版本，含基础库及主要间接依赖。 |
| `requirements.txt` | 基础直接依赖与范围，基础锁文件缺失时使用；含 PySide6、转录、录音和说话人依赖。 |
| `requirements-gpu.lock.txt` | `setup-gpu.bat` 优先使用的 NVIDIA 运行库固定版本。 |
| `requirements-gpu.txt` | GPU 运行库范围，GPU 锁文件缺失时使用。 |
| `requirements-sensevoice-gpu.txt` | SenseVoice GPU 脚本从官方来源下载固定的 `sherpa-onnx==1.13.8+cuda12.cudnn9`。 |
| `setup-speakers.bat` | 确保 sherpa-onnx 已安装并下载说话人模型。 |
| `.env.example` / `settings.py` | 空白模板与程序默认配置；安装脚本仅在 `.env` 缺失时复制模板。 |

SenseVoice GPU 脚本先下载，再移除文件冲突的 CPU `sherpa-onnx-core` 并安装 CUDA 版。CUDA 版也支持 CPU 推理，之后基础安装不会将其降回 CPU 版。不要用含 GPU 依赖的 `pip freeze` 覆盖基础锁文件。

开发时在项目目录检查依赖并运行自动测试（接口使用模拟响应，不调用收费 API）：

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

常见处理：无法录音先做硬件测试、检查麦克风权限；缺 CUDA DLL 时退出程序并重跑对应 GPU 脚本；模型下载失败检查网络后重试；Notion 404 检查目标页面是否共享给连接。具体错误见「运行信息」。
