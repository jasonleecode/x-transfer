import sys
import os
import time
import numpy as np
import sounddevice as sd
from qam_modulation import QAMModulator
from scipy.signal import resample
import threading
from functools import wraps
from tqdm import tqdm
import hashlib
import struct


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

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time} seconds to execute.")
        return result
    return wrapper

def generate_sync_signal(duration=1.0, frequency=1000):
    """生成同步信号"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)

def generate_file_header(file_size, file_hash, protocol):
    """生成文件头信息"""
    header = struct.pack('>I', file_size)  # 文件大小（4字节）
    header += file_hash.encode()[:16]  # 文件哈希（16字节）
    header += protocol.encode()[:8]  # 协议类型（8字节）
    return header

def calculate_file_hash(file_path):
    """计算文件MD5哈希"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def safe_audio_play(signal, samplerate):
    """安全的音频播放，带错误处理"""
    try:
        sd.play(signal, samplerate)
        sd.wait()
        return True
    except Exception as e:
        print(f"音频播放错误: {e}")
        return False

def generate_tone(frequency, duration, fs):
    """生成给定频率的音频信号"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)

def fsk_encode_bit(bit):
    """将单个比特编码为音频信号"""
    return generate_tone(f1 if bit == '1' else f0, duration, fs)

def file_to_bits(file_path):
    """将文件内容转换为比特字符串"""
    with open(file_path, 'rb') as f:
        byte_data = f.read()
    # 将每个字节转换为8位二进制字符串
    bit_string = ''.join(format(byte, '08b') for byte in byte_data)
    return bit_string

# 发送方式一：一股脑发送，大文件容易卡死
def send_data_raw(data):
    """将完整数据编码为音频并发送"""
    signal = np.concatenate([
        generate_tone(f1 if bit == '1' else f0, duration, fs)
        for bit in data
    ])
    sd.play(signal, fs)
    sd.wait()
    
# 发送方式二：分块发送
def send_data_in_chunks(data):
    """分块发送比特数据"""
    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i:i + CHUNK_SIZE]
        signal = np.concatenate([fsk_encode_bit(bit) for bit in chunk])
        sd.play(signal, fs)
        sd.wait()

# 发送方式三：多线程方式
def send_data_by_thread(data):
    print("--")

@timing_decorator
def send_file_ask(filename):
    """将文件内容转换为 ASK 信号并发送"""
    if not os.path.exists(filename):
        print(f"File '{filename}' not found!")
        sys.exit(1)

    # 读取文件并转换为比特流
    with open(filename, "rb") as f:
        byte_data = f.read()
    bit_data = [int(bit) for byte in byte_data for bit in f"{byte:08b}"]
    
    # ASK调制：1用高幅度，0用低幅度
    signal = np.concatenate([
        generate_tone(f0, duration, fs) * (0.1 if bit == 0 else 1.0)
        for bit in bit_data
    ])
    
    sd.play(signal, fs)
    sd.wait()
    
    print("File sent successfully!")
    
@timing_decorator
def send_file_fsk(filename):
    """将文件内容转换为 FSK 信号并发送"""
    if not os.path.exists(filename):
        print(f"File '{filename}' not found!")
        sys.exit(1)

    # 读取文件并转换为比特流
    with open(filename, "rb") as f:
        byte_data = f.read()
    bit_data = [int(bit) for byte in byte_data for bit in f"{byte:08b}"]
    send_data_raw(bit_data)
    
    print("File sent successfully!")

@timing_decorator
def send_file_qam(filename, chunk_size=1024):
    """将文件内容转换为 QAM 信号并发送"""
    try:
        if not os.path.exists(filename):
            print(f"File '{filename}' not found!")
            return False

        # 计算文件信息
        file_size = os.path.getsize(filename)
        file_hash = calculate_file_hash(filename)
        
        print(f"文件大小: {file_size} 字节")
        print(f"文件哈希: {file_hash}")

        # 生成文件头
        header = generate_file_header(file_size, file_hash, "QAM")
        header_bits = [int(bit) for byte in header for bit in f"{byte:08b}"]

        # 读取文件并转换为比特流
        with open(filename, "rb") as f:
            byte_data = f.read()
        bit_data = [int(bit) for byte in byte_data for bit in f"{byte:08b}"]

        # 组合头部和数据
        all_bits = header_bits + bit_data

        # 调制为 QAM 信号
        modulator = QAMModulator(M, carrier_freq, symbol_rate, sampling_rate)
        
        # 发送同步信号
        print("发送同步信号...")
        sync_signal = generate_sync_signal(2.0, 1000)
        if not safe_audio_play(sync_signal, sampling_rate):
            return False
        
        # 发送文件头
        print("发送文件头...")
        header_signal = modulator.modulate_qam(header_bits)
        if not safe_audio_play(header_signal, sampling_rate):
            return False
        
        # 发送文件数据
        print("发送文件数据...")
        with tqdm(total=len(bit_data), desc="Sending file", unit="bit") as pbar:
            for i in range(0, len(bit_data), chunk_size):
                chunk = bit_data[i:i + chunk_size]  # 分块
                signal_chunk = modulator.modulate_qam(chunk)  # 调制
                if not safe_audio_play(signal_chunk, sampling_rate):
                    return False
                pbar.update(len(chunk))  # 更新进度

        print("File sent successfully!")
        return True
        
    except Exception as e:
        print(f"发送文件时发生错误: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python send_file.py <file_path> <protocol>")
        print("Protocol can be 'qam' or 'fsk'")
        sys.exit(1)

    file_path = sys.argv[1]
    protocol = sys.argv[2].lower()  # 将协议转换为小写以保持一致性

    if protocol not in ['qam', 'fsk', 'ask']:
        print("Invalid protocol. Please choose 'qam'/'fsk'/'ask'.")
        sys.exit(1)

    print(f"Sending file: {file_path} using {protocol.upper()} protocol")

    if protocol == 'qam':
        send_file_qam(file_path)
    elif protocol == 'fsk':
        send_file_fsk(file_path)
    elif protocol == 'ask':
        send_file_ask(file_path)
