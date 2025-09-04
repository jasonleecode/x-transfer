import numpy as np
import sounddevice as sd
from scipy.fft import fft
import sys
from qam_modulation import QAMModulator
import struct
import hashlib
import time
from realtime_qam_receiver import RealtimeQAMReceiver

# ask & fsk parameter
CHUNK_SIZE = 1024
fs = 44100  # 采样率
duration = 0.01  # 每个比特的持续时间
f0 = 500  # 频率表示 0
f1 = 1000  # 频率表示 1

# QAM parameter
M = 16  # 使用 16-QAM
symbol_rate = 1000  # 符号率（symbols per second）
carrier_freq = 2000  # 载波频率（Hz）
sampling_rate = 44100  # 采样率（Hz）


def decode_tone(signal):
    """通过FFT解码音频信号为比特"""
    N = len(signal)
    yf = np.abs(fft(signal))
    xf = np.fft.fftfreq(N, 1 / fs)
    peak_freq = abs(xf[np.argmax(yf)])
    return '1' if abs(peak_freq - f1) < 50 else '0'

def receive_bit():
    """接收一个比特并解码"""
    signal = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return decode_tone(signal[:, 0])

def receive_data(num_bits):
    """接收指定数量的比特并返回二进制字符串"""
    bit_string = ''.join(receive_bit() for _ in range(num_bits))
    return bit_string

def bits_to_file(bit_string, output_file):
    """将比特字符串转换为文件并保存"""
    try:
        byte_data = int(bit_string, 2).to_bytes(len(bit_string) // 8, byteorder='big')
        with open(output_file, 'wb') as f:
            f.write(byte_data)
        return True
    except Exception as e:
        print(f"保存文件时发生错误: {e}")
        return False

def detect_sync_signal(signal, threshold=0.3, frequency=1000):
    """检测同步信号"""
    from scipy.fft import fft, fftfreq
    
    # 使用滑动窗口检测同步信号
    window_size = int(fs * 0.5)  # 0.5秒窗口
    step_size = int(fs * 0.1)    # 0.1秒步长
    
    for i in range(0, len(signal) - window_size, step_size):
        window = signal[i:i + window_size]
        
        # FFT分析
        fft_result = np.abs(fft(window))
        freqs = fftfreq(len(window), 1/fs)
        
        # 检查目标频率附近是否有强信号
        target_freq_idx = np.argmin(np.abs(freqs - frequency))
        if fft_result[target_freq_idx] > threshold * np.max(fft_result):
            return i + window_size  # 返回同步信号结束位置
    
    return None

def parse_file_header(header_bits):
    """解析文件头信息"""
    try:
        # 转换为字节
        header_bytes = int(''.join(map(str, header_bits)), 2).to_bytes(len(header_bits) // 8, byteorder='big')
        
        # 解析文件大小
        file_size = struct.unpack('>I', header_bytes[:4])[0]
        
        # 解析文件哈希
        file_hash = header_bytes[4:20].decode().strip('\x00')
        
        # 解析协议类型
        protocol = header_bytes[20:28].decode().strip('\x00')
        
        return file_size, file_hash, protocol
    except Exception as e:
        print(f"解析文件头时发生错误: {e}")
        return None, None, None

def safe_audio_record(duration, samplerate):
    """安全的音频录制，带错误处理"""
    try:
        signal = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()
        return signal[:, 0]
    except Exception as e:
        print(f"音频录制错误: {e}")
        return None

def receive_file_qam_realtime(output_filename):
    """实时接收QAM文件（边接收边解析）"""
    print("使用实时模式接收QAM文件...")
    
    # 创建实时接收器
    receiver = RealtimeQAMReceiver()
    
    def progress_callback(progress, received_bits=0, expected_bits=0):
        if expected_bits > 0:
            print(f"\r接收进度: {progress:.1f}% ({received_bits}/{expected_bits} 比特)", end='', flush=True)
        else:
            print(f"\r接收进度: {progress:.1f}%", end='', flush=True)
    
    receiver.set_progress_callback(progress_callback)
    
    # 开始接收
    success = receiver.start_receiving(output_filename)
    
    if success:
        print("\n文件接收成功!")
        return True
    else:
        print("\n文件接收失败!")
        return False
        
def receive_file_qam(output_filename, duration=30):
    """接收QAM文件"""
    try:
        print("正在监听传入数据...")
        recorded_signal = safe_audio_record(duration, sampling_rate)
        
        if recorded_signal is None:
            print("录音失败")
            return False

        # 检测同步信号
        print("检测同步信号...")
        sync_end = detect_sync_signal(recorded_signal)
        
        if sync_end is None:
            print("未检测到同步信号")
            return False
        
        print(f"检测到同步信号，开始位置: {sync_end}")
        
        # 解调 QAM 信号
        modulator = QAMModulator(M, carrier_freq, symbol_rate, sampling_rate)
        
        # 接收文件头
        print("接收文件头...")
        header_duration = 2.0  # 文件头持续时间
        header_start = sync_end
        header_end = header_start + int(header_duration * sampling_rate)
        header_signal = recorded_signal[header_start:header_end]
        header_bits = modulator.demodulate_qam(header_signal)
        
        # 解析文件头
        file_size, file_hash, protocol = parse_file_header(header_bits)
        
        if file_size is None:
            print("文件头解析失败")
            return False
        
        print(f"文件大小: {file_size} 字节")
        print(f"文件哈希: {file_hash}")
        print(f"协议类型: {protocol}")
        
        # 接收文件数据
        print("接收文件数据...")
        data_start = header_end
        data_duration = (file_size * 8) / (symbol_rate * modulator.k)  # 估算数据持续时间
        data_end = data_start + int(data_duration * sampling_rate)
        data_signal = recorded_signal[data_start:data_end]
        bit_data = modulator.demodulate_qam(data_signal)
        
        # 截取到正确的长度
        expected_bits = file_size * 8
        if len(bit_data) > expected_bits:
            bit_data = bit_data[:expected_bits]
        
        # 将比特流还原为字节并写入文件
        byte_data = bytearray()
        for i in range(0, len(bit_data), 8):
            if i + 8 <= len(bit_data):
                byte_bits = bit_data[i:i+8]
                byte_value = int("".join(map(str, byte_bits)), 2)
                byte_data.append(byte_value)
        
        # 保存文件
        if bits_to_file(bit_data, output_filename):
            print(f"文件接收并保存为: {output_filename}")
            
            # 验证文件哈希
            received_hash = hashlib.md5(byte_data).hexdigest()
            if received_hash == file_hash:
                print("文件完整性验证成功")
            else:
                print(f"文件完整性验证失败: 期望 {file_hash}, 实际 {received_hash}")
            
            return True
        else:
            print("文件保存失败")
            return False
            
    except Exception as e:
        print(f"接收文件时发生错误: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python receive_file.py <output_file> [protocol] [mode]")
        print("Protocol can be 'qam', 'fsk', or 'ask'")
        print("Mode can be 'realtime' or 'batch' (default: batch)")
        sys.exit(1)

    output_file = sys.argv[1]
    protocol = sys.argv[2].lower() if len(sys.argv) > 2 else 'qam'
    mode = sys.argv[3].lower() if len(sys.argv) > 3 else 'batch'
    
    print(f"使用 {protocol.upper()} 协议接收文件...")
    print(f"模式: {'实时' if mode == 'realtime' else '批量'}")
    
    if protocol == 'qam':
        if mode == 'realtime':
            success = receive_file_qam_realtime(output_file)
        else:
            success = receive_file_qam(output_file)
        
        if success:
            print("文件接收成功!")
        else:
            print("文件接收失败!")
    elif protocol == 'fsk':
        print("FSK协议接收功能待实现")
    elif protocol == 'ask':
        print("ASK协议接收功能待实现")
    else:
        print("不支持的协议类型")

