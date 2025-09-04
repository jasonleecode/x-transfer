# x-transfer

通过音频信号传输数据的项目，支持多种调制方式。

## 功能特性

- **多种调制方式**：支持QAM、FSK、ASK、摩尔斯电码、FT8
- **文件传输**：支持任意文件的音频传输
- **同步机制**：内置同步信号确保传输可靠性
- **错误处理**：完善的错误检测和异常处理
- **文件完整性**：MD5哈希验证确保文件完整性
- **进度显示**：实时显示传输进度

## 支持的调制方式

### 1. QAM (Quadrature Amplitude Modulation)
- 支持4-QAM、16-QAM、64-QAM、256-QAM
- 高数据率传输
- 适合大文件传输

### 2. FSK (Frequency Shift Keying)
- 简单可靠的调制方式
- 适合小文件传输

### 3. ASK (Amplitude Shift Keying)
- 基于幅度调制的简单方式
- 适合文本传输

### 4. 摩尔斯电码 (Morse Code)
- 经典的数字通信方式
- 适合文本消息传输

### 5. FT8
- 基于WSJT-X的现代数字通信协议
- 适合业余无线电通信

### 6. OFDM + 自适应QAM
- 正交频分复用技术
- 自适应调制编码
- 多子载波并行传输
- 高频谱效率和高可靠性

### 7. 高级OFDM
- 多天线MIMO支持
- 预编码和波束成形
- 信道编码和交织
- 高级同步算法
- 自适应均衡

### 8. 双声道传输
- 左右声道并行传输
- 多种传输模式（并行、备份、混合）
- 自动同步检测
- 错误检测和恢复
- 理论上双倍传输效率

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 发送文件

```bash
# 使用QAM协议发送文件
python send_fine_bysnd.py <file_path> qam

# 使用FSK协议发送文件
python send_fine_bysnd.py <file_path> fsk

# 使用ASK协议发送文件
python send_fine_bysnd.py <file_path> ask
```

### 接收文件

```bash
# 使用QAM协议接收文件（批量模式）
python receive_file_bysnd.py <output_file> qam batch

# 使用QAM协议实时接收文件（推荐）
python receive_file_bysnd.py <output_file> qam realtime

# 使用FSK协议接收文件
python receive_file_bysnd.py <output_file> fsk

# 使用ASK协议接收文件
python receive_file_bysnd.py <output_file> ask

# 直接使用实时QAM接收器
python realtime_qam_receiver.py <output_file>
```

### 接收模式说明

- **实时模式 (realtime)**：边接收边解析，适合长文件传输，内存占用低
- **批量模式 (batch)**：先录制完整音频再解析，适合短文件传输

### 摩尔斯电码

```python
from morse_modulation import MorseModulator

modulator = MorseModulator()
modulator.send_text("HELLO WORLD")
```

### FT8通信

```python
from ft8_modulation import FT8Modulator

modulator = FT8Modulator()
modulator.send_ft8_message("HELLO WORLD", "CQ", "TEST", "FN42")
```

### OFDM自适应QAM

```python
from ofdm_adaptive_qam import OFDMAdaptiveQAM, ModulationType

# 创建OFDM调制器
ofdm_mod = OFDMAdaptiveQAM(
    sampling_rate=44100,
    num_subcarriers=32,
    num_pilot_carriers=4
)

# 调制
bits = [1, 0, 1, 1, 0, 1, 0, 0]
signal = ofdm_mod.modulate_with_carrier(bits)

# 解调
demodulated_bits, used_mod = ofdm_mod.demodulate_with_carrier(signal)
```

### 高级OFDM

```python
from advanced_ofdm import AdvancedOFDM, OFDMConfig

# 创建配置
config = OFDMConfig()
config.num_subcarriers = 32
config.use_fec = True
config.use_sync = True

# 创建高级OFDM调制器
ofdm_mod = AdvancedOFDM(config)

# 调制和解调
bits = [1, 0, 1, 1, 0, 1, 0, 0]
signal = ofdm_mod.modulate(bits, modulation_order=16, use_advanced_features=True)
demodulated_bits, used_mod = ofdm_mod.demodulate(signal, expected_modulation_order=16)
```

### 性能基准测试

```python
from ofdm_benchmark import OFDMBenchmark

# 运行基准测试
benchmark = OFDMBenchmark()
results = benchmark.run_all_benchmarks()
benchmark.generate_report()
```

### 双声道传输

```python
from stereo_transmission import StereoTransmission, StereoMode

# 创建双声道传输器
stereo_transmission = StereoTransmission(
    sampling_rate=44100,
    carrier_freq=2000,
    symbol_rate=1000,
    modulation_order=16,
    mode=StereoMode.PARALLEL
)

# 传输数据
bits = [1, 0, 1, 1, 0, 1, 0, 0]
stereo_transmission.transmit_data(bits)
received_bits = stereo_transmission.receive_data(duration=10)
```

### 双声道文件传输

```bash
# 发送文件
python stereo_cli.py send file.txt --mode parallel

# 接收文件
python stereo_cli.py receive output.txt --mode parallel

# 运行测试
python stereo_cli.py test --audio-test

# 基准测试
python stereo_cli.py benchmark
```

## 技术参数

- **采样率**：44.1 kHz
- **QAM载波频率**：2 kHz
- **QAM符号率**：1 kbps
- **FSK频率**：500 Hz (0), 1000 Hz (1)
- **摩尔斯电码频率**：1 kHz

## 注意事项

1. 确保音频设备正常工作
2. 发送和接收端使用相同的协议
3. 保持环境相对安静以减少干扰
4. 大文件传输可能需要较长时间

## 项目结构

```
x-transfer/
├── qam_modulation.py           # QAM调制器
├── morse_modulation.py         # 摩尔斯电码调制器
├── ft8_modulation.py           # FT8调制器
├── ofdm_adaptive_qam.py        # OFDM自适应QAM调制器
├── advanced_ofdm.py            # 高级OFDM调制器
├── ofdm_benchmark.py           # OFDM性能基准测试
├── ofdm_examples.py            # OFDM使用示例
├── stereo_transmission.py      # 双声道传输模块
├── stereo_cli.py               # 双声道传输命令行工具
├── test_stereo.py              # 双声道传输测试
├── realtime_qam_receiver.py    # 实时QAM接收器
├── send_fine_bysnd.py          # 发送端
├── receive_file_bysnd.py       # 接收端
├── test_realtime.py            # 实时接收测试
├── requirements.txt            # 依赖包
└── README.md                  # 说明文档
```

## 开发计划

- [ ] 实现视频传输功能
- [ ] 添加更多调制方式
- [ ] 优化传输算法
- [ ] 添加GUI界面
- [ ] 支持网络传输

