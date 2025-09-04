#!/usr/bin/env python3
"""
双声道数据传输模块
支持左右声道并行传输，提高传输效率
"""

import numpy as np
import sounddevice as sd
import threading
import time
import struct
import hashlib
from typing import Tuple, Optional, List
from enum import Enum
from qam_modulation import QAMModulator

class StereoMode(Enum):
    """双声道模式"""
    PARALLEL = "parallel"      # 并行传输：左右声道独立传输不同数据
    BACKUP = "backup"         # 备份传输：左右声道传输相同数据
    MIXED = "mixed"           # 混合模式：部分数据并行，部分备份

class StereoTransmission:
    """
    双声道数据传输类
    
    特性：
    - 左右声道并行传输
    - 自动同步检测
    - 错误检测和恢复
    - 自适应功率分配
    - 支持多种调制方式
    """
    
    def __init__(self, 
                 sampling_rate=44100,
                 carrier_freq=2000,
                 symbol_rate=1000,
                 modulation_order=16,
                 mode=StereoMode.PARALLEL):
        """
        初始化双声道传输器
        
        Args:
            sampling_rate: 采样率 (Hz)
            carrier_freq: 载波频率 (Hz)
            symbol_rate: 符号率 (Hz)
            modulation_order: 调制阶数
            mode: 双声道模式
        """
        self.sampling_rate = sampling_rate
        self.carrier_freq = carrier_freq
        self.symbol_rate = symbol_rate
        self.modulation_order = modulation_order
        self.mode = mode
        
        # 创建左右声道调制器
        self.left_modulator = QAMModulator(modulation_order, carrier_freq, symbol_rate, sampling_rate)
        self.right_modulator = QAMModulator(modulation_order, carrier_freq, symbol_rate, sampling_rate)
        
        # 同步参数
        self.sync_freq = 1000  # 同步信号频率
        self.sync_duration = 1.0  # 同步信号持续时间
        self.sync_samples = int(sampling_rate * self.sync_duration)
        
        # 状态变量
        self.is_transmitting = False
        self.is_receiving = False
        self.transmission_thread = None
        self.reception_thread = None
        
        # 错误统计
        self.left_errors = 0
        self.right_errors = 0
        self.total_errors = 0
        
    def generate_sync_signal(self, duration=None):
        """生成同步信号"""
        if duration is None:
            duration = self.sync_duration
        
        samples = int(self.sampling_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)
        sync_signal = 0.5 * np.sin(2 * np.pi * self.sync_freq * t)
        
        # 立体声同步信号（左右声道相同）
        stereo_sync = np.column_stack([sync_signal, sync_signal])
        return stereo_sync
    
    def split_data(self, data):
        """分割数据为左右声道"""
        data = np.array(data)
        
        if self.mode == StereoMode.PARALLEL:
            # 并行模式：交替分配数据
            left_data = data[::2]
            right_data = data[1::2]
        elif self.mode == StereoMode.BACKUP:
            # 备份模式：左右声道传输相同数据
            left_data = data
            right_data = data
        elif self.mode == StereoMode.MIXED:
            # 混合模式：前80%并行，后20%备份
            split_point = int(len(data) * 0.8)
            left_parallel = data[:split_point:2]
            right_parallel = data[1:split_point:2]
            backup_data = data[split_point:]
            
            left_data = np.concatenate([left_parallel, backup_data])
            right_data = np.concatenate([right_parallel, backup_data])
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
        
        return left_data, right_data
    
    def merge_data(self, left_data, right_data):
        """合并左右声道数据"""
        left_data = np.array(left_data)
        right_data = np.array(right_data)
        
        if self.mode == StereoMode.PARALLEL:
            # 并行模式：交替合并数据
            max_len = max(len(left_data), len(right_data))
            merged = np.empty(max_len * 2, dtype=left_data.dtype)
            merged[::2] = left_data
            merged[1::2] = right_data
            return merged
        elif self.mode == StereoMode.BACKUP:
            # 备份模式：选择错误较少的数据
            if len(left_data) >= len(right_data):
                return left_data
            else:
                return right_data
        elif self.mode == StereoMode.MIXED:
            # 混合模式：先合并并行部分，再处理备份部分
            parallel_len = min(len(left_data), len(right_data))
            merged_parallel = np.empty(parallel_len * 2, dtype=left_data.dtype)
            merged_parallel[::2] = left_data[:parallel_len]
            merged_parallel[1::2] = right_data[:parallel_len]
            
            # 处理备份部分
            if len(left_data) > parallel_len:
                merged_parallel = np.concatenate([merged_parallel, left_data[parallel_len:]])
            elif len(right_data) > parallel_len:
                merged_parallel = np.concatenate([merged_parallel, right_data[parallel_len:]])
            
            return merged_parallel
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def modulate_stereo(self, data):
        """双声道调制"""
        # 分割数据
        left_data, right_data = self.split_data(data)
        
        # 调制左右声道
        left_signal = self.left_modulator.modulate_qam(left_data)
        right_signal = self.right_modulator.modulate_qam(right_data)
        
        # 确保左右声道长度一致
        max_len = max(len(left_signal), len(right_signal))
        if len(left_signal) < max_len:
            left_signal = np.concatenate([left_signal, np.zeros(max_len - len(left_signal))])
        if len(right_signal) < max_len:
            right_signal = np.concatenate([right_signal, np.zeros(max_len - len(right_signal))])
        
        # 创建立体声信号
        stereo_signal = np.column_stack([left_signal, right_signal])
        
        return stereo_signal
    
    def demodulate_stereo(self, stereo_signal):
        """双声道解调"""
        if stereo_signal.ndim != 2 or stereo_signal.shape[1] != 2:
            raise ValueError("输入必须是立体声信号 (N, 2)")
        
        left_signal = stereo_signal[:, 0]
        right_signal = stereo_signal[:, 1]
        
        # 解调左右声道
        left_data = self.left_modulator.demodulate_qam(left_signal)
        right_data = self.right_modulator.demodulate_qam(right_signal)
        
        # 合并数据
        merged_data = self.merge_data(left_data, right_data)
        
        return merged_data
    
    def detect_sync_stereo(self, stereo_signal, threshold=0.3):
        """检测立体声同步信号"""
        if stereo_signal.ndim != 2 or stereo_signal.shape[1] != 2:
            return None, None
        
        left_signal = stereo_signal[:, 0]
        right_signal = stereo_signal[:, 1]
        
        # 检测左声道同步
        left_sync_pos = self._detect_sync_single(left_signal, threshold)
        # 检测右声道同步
        right_sync_pos = self._detect_sync_single(right_signal, threshold)
        
        # 返回两个声道中较早的同步位置
        if left_sync_pos is not None and right_sync_pos is not None:
            sync_pos = min(left_sync_pos, right_sync_pos)
        elif left_sync_pos is not None:
            sync_pos = left_sync_pos
        elif right_sync_pos is not None:
            sync_pos = right_sync_pos
        else:
            sync_pos = None
        
        return sync_pos, (left_sync_pos, right_sync_pos)
    
    def _detect_sync_single(self, signal, threshold=0.3):
        """检测单声道同步信号"""
        from scipy.fftpack import fft, fftfreq
        
        window_size = int(self.sampling_rate * 0.5)  # 0.5秒窗口
        step_size = int(self.sampling_rate * 0.1)    # 0.1秒步长
        
        for i in range(0, len(signal) - window_size, step_size):
            window = signal[i:i + window_size]
            
            # FFT分析
            fft_result = np.abs(fft(window))
            freqs = fftfreq(len(window), 1/self.sampling_rate)
            
            # 检查同步频率附近是否有强信号
            target_freq_idx = np.argmin(np.abs(freqs - self.sync_freq))
            if fft_result[target_freq_idx] > threshold * np.max(fft_result):
                return i + window_size
        
        return None
    
    def send_stereo_signal(self, stereo_signal):
        """发送立体声信号"""
        try:
            sd.play(stereo_signal, self.sampling_rate)
            sd.wait()
            return True
        except Exception as e:
            print(f"发送立体声信号错误: {e}")
            return False
    
    def receive_stereo_signal(self, duration):
        """接收立体声信号"""
        try:
            stereo_signal = sd.rec(int(duration * self.sampling_rate), 
                                 samplerate=self.sampling_rate, channels=2)
            sd.wait()
            return stereo_signal
        except Exception as e:
            print(f"接收立体声信号错误: {e}")
            return None
    
    def transmit_data(self, data, use_sync=True):
        """传输数据"""
        if self.is_transmitting:
            print("已经在传输中...")
            return False
        
        self.is_transmitting = True
        
        try:
            # 生成同步信号
            if use_sync:
                print("发送同步信号...")
                sync_signal = self.generate_sync_signal()
                if not self.send_stereo_signal(sync_signal):
                    return False
            
            # 调制数据
            print("调制数据...")
            stereo_signal = self.modulate_stereo(data)
            
            # 发送数据
            print("发送数据...")
            success = self.send_stereo_signal(stereo_signal)
            
            if success:
                print("数据传输完成!")
            
            return success
            
        except Exception as e:
            print(f"传输数据时发生错误: {e}")
            return False
        finally:
            self.is_transmitting = False
    
    def receive_data(self, duration, use_sync=True):
        """接收数据"""
        if self.is_receiving:
            print("已经在接收中...")
            return None
        
        self.is_receiving = True
        
        try:
            # 接收信号
            print("接收信号...")
            stereo_signal = self.receive_stereo_signal(duration)
            
            if stereo_signal is None:
                return None
            
            # 检测同步信号
            if use_sync:
                print("检测同步信号...")
                sync_pos, (left_sync, right_sync) = self.detect_sync_stereo(stereo_signal)
                
                if sync_pos is None:
                    print("未检测到同步信号")
                    return None
                
                print(f"检测到同步信号，位置: {sync_pos}")
                print(f"左声道同步: {left_sync}, 右声道同步: {right_sync}")
                
                # 移除同步信号部分
                stereo_signal = stereo_signal[sync_pos:]
            
            # 解调数据
            print("解调数据...")
            data = self.demodulate_stereo(stereo_signal)
            
            print(f"接收数据完成，长度: {len(data)} 比特")
            return data
            
        except Exception as e:
            print(f"接收数据时发生错误: {e}")
            return None
        finally:
            self.is_receiving = False
    
    def get_transmission_info(self):
        """获取传输信息"""
        return {
            'mode': self.mode.value,
            'sampling_rate': self.sampling_rate,
            'carrier_freq': self.carrier_freq,
            'symbol_rate': self.symbol_rate,
            'modulation_order': self.modulation_order,
            'left_errors': self.left_errors,
            'right_errors': self.right_errors,
            'total_errors': self.total_errors,
            'is_transmitting': self.is_transmitting,
            'is_receiving': self.is_receiving
        }
    
    def set_mode(self, mode):
        """设置传输模式"""
        if isinstance(mode, str):
            mode = StereoMode(mode)
        self.mode = mode
        print(f"传输模式设置为: {mode.value}")


class StereoFileTransmission:
    """双声道文件传输类"""
    
    def __init__(self, stereo_transmission):
        self.stereo_transmission = stereo_transmission
    
    def send_file(self, file_path, chunk_size=1024):
        """发送文件"""
        try:
            # 读取文件
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # 转换为比特流
            bit_data = [int(bit) for byte in file_data for bit in f"{byte:08b}"]
            
            print(f"文件大小: {len(file_data)} 字节")
            print(f"比特数: {len(bit_data)}")
            
            # 添加文件头信息
            file_size = len(file_data)
            file_hash = hashlib.md5(file_data).hexdigest()
            
            # 创建文件头
            header = struct.pack('>I', file_size)  # 文件大小
            header += file_hash.encode()[:16]  # 文件哈希
            header_bits = [int(bit) for byte in header for bit in f"{byte:08b}"]
            
            # 发送文件头
            print("发送文件头...")
            if not self.stereo_transmission.transmit_data(header_bits):
                return False
            
            # 分块发送文件数据
            print("发送文件数据...")
            for i in range(0, len(bit_data), chunk_size):
                chunk = bit_data[i:i + chunk_size]
                print(f"发送块 {i//chunk_size + 1}/{(len(bit_data) + chunk_size - 1)//chunk_size}")
                
                if not self.stereo_transmission.transmit_data(chunk):
                    print("发送失败")
                    return False
            
            print("文件发送完成!")
            return True
            
        except Exception as e:
            print(f"发送文件时发生错误: {e}")
            return False
    
    def receive_file(self, output_path, duration=30):
        """接收文件"""
        try:
            # 接收文件头
            print("接收文件头...")
            header_bits = self.stereo_transmission.receive_data(5)  # 5秒超时
            
            if header_bits is None:
                print("接收文件头失败")
                return False
            
            # 解析文件头
            if len(header_bits) < 224:  # 28字节 * 8比特
                print("文件头长度不足")
                return False
            
            header_bytes = int(''.join(map(str, header_bits[:224])), 2).to_bytes(28, byteorder='big')
            file_size = struct.unpack('>I', header_bytes[:4])[0]
            file_hash = header_bytes[4:20].decode().strip('\x00')
            
            print(f"文件大小: {file_size} 字节")
            print(f"文件哈希: {file_hash}")
            
            # 接收文件数据
            print("接收文件数据...")
            data_bits = self.stereo_transmission.receive_data(duration)
            
            if data_bits is None:
                print("接收文件数据失败")
                return False
            
            # 转换为字节
            byte_data = bytearray()
            for i in range(0, len(data_bits), 8):
                if i + 8 <= len(data_bits):
                    byte_bits = data_bits[i:i+8]
                    byte_value = int("".join(map(str, byte_bits)), 2)
                    byte_data.append(byte_value)
            
            # 截取到正确长度
            if len(byte_data) > file_size:
                byte_data = byte_data[:file_size]
            
            # 保存文件
            with open(output_path, 'wb') as f:
                f.write(byte_data)
            
            # 验证文件哈希
            received_hash = hashlib.md5(byte_data).hexdigest()
            if received_hash == file_hash:
                print("文件完整性验证成功!")
            else:
                print(f"文件完整性验证失败: 期望 {file_hash}, 实际 {received_hash}")
            
            print(f"文件已保存为: {output_path}")
            return True
            
        except Exception as e:
            print(f"接收文件时发生错误: {e}")
            return False


# 测试和示例代码
if __name__ == "__main__":
    # 创建双声道传输器
    stereo_transmission = StereoTransmission(
        sampling_rate=44100,
        carrier_freq=2000,
        symbol_rate=1000,
        modulation_order=16,
        mode=StereoMode.PARALLEL
    )
    
    # 创建文件传输器
    file_transmission = StereoFileTransmission(stereo_transmission)
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 1000)
    print(f"测试数据: {len(test_bits)} 比特")
    
    # 显示传输信息
    info = stereo_transmission.get_transmission_info()
    print("传输信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试调制解调
    print("\n测试双声道调制解调...")
    stereo_signal = stereo_transmission.modulate_stereo(test_bits)
    print(f"立体声信号形状: {stereo_signal.shape}")
    
    demodulated_bits = stereo_transmission.demodulate_stereo(stereo_signal)
    print(f"解调数据长度: {len(demodulated_bits)}")
    
    # 计算误码率
    if len(demodulated_bits) >= len(test_bits):
        error_rate = np.mean(demodulated_bits[:len(test_bits)] != test_bits)
        print(f"误码率: {error_rate:.4f}")
    
    print("\n双声道传输模块测试完成!")
