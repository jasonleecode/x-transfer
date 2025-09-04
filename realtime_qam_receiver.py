import numpy as np
import sounddevice as sd
import threading
import queue
import time
import struct
import hashlib
from qam_modulation import QAMModulator
from scipy.fft import fft, fftfreq
from collections import deque
import os

class RealtimeQAMReceiver:
    """
    实时QAM接收器 - 边接收边解析
    """
    def __init__(self, sampling_rate=44100, symbol_rate=1000, carrier_freq=2000, M=16):
        self.sampling_rate = sampling_rate
        self.symbol_rate = symbol_rate
        self.carrier_freq = carrier_freq
        self.M = M
        
        # 初始化QAM调制器
        self.modulator = QAMModulator(M, carrier_freq, symbol_rate, sampling_rate)
        self.samples_per_symbol = int(sampling_rate / symbol_rate)
        
        # 音频流参数
        self.chunk_size = 1024  # 每次处理的样本数
        self.buffer_size = 10   # 缓冲区大小（秒）
        self.max_buffer_samples = int(sampling_rate * self.buffer_size)
        
        # 线程控制
        self.is_recording = False
        self.is_processing = False
        self.audio_thread = None
        self.processing_thread = None
        
        # 数据缓冲区
        self.audio_buffer = deque(maxlen=self.max_buffer_samples)
        self.symbol_buffer = deque(maxlen=1000)  # 符号缓冲区
        
        # 同步检测
        self.sync_detected = False
        self.sync_position = 0
        self.header_received = False
        self.file_size = 0
        self.file_hash = ""
        self.protocol = ""
        
        # 文件接收状态
        self.received_bits = []
        self.expected_bits = 0
        self.file_complete = False
        
        # 进度跟踪
        self.progress_callback = None
        self.start_time = None
        self.last_progress_time = 0
        
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
        
    def audio_callback(self, indata, frames, time, status):
        """音频流回调函数"""
        if status:
            print(f"音频流状态: {status}")
        
        if self.is_recording:
            # 将音频数据添加到缓冲区
            self.audio_buffer.extend(indata[:, 0])
    
    def detect_sync_signal_realtime(self, signal_chunk, threshold=0.3):
        """实时检测同步信号"""
        if len(signal_chunk) < int(self.sampling_rate * 0.5):
            return False, 0
            
        # 使用滑动窗口检测
        window_size = int(self.sampling_rate * 0.5)  # 0.5秒窗口
        step_size = int(self.sampling_rate * 0.05)   # 0.05秒步长，提高精度
        
        max_energy = 0
        best_position = 0
        
        for i in range(0, len(signal_chunk) - window_size, step_size):
            window = signal_chunk[i:i + window_size]
            
            # FFT分析
            fft_result = np.abs(fft(window))
            freqs = fftfreq(len(window), 1/self.sampling_rate)
            
            # 检查1000Hz附近是否有强信号
            target_freq_idx = np.argmin(np.abs(freqs - 1000))
            energy = fft_result[target_freq_idx]
            
            # 检查信号是否足够强且持续时间足够长
            if energy > threshold * np.max(fft_result):
                # 检查后续窗口是否也有强信号（确保是持续的同步信号）
                if i + window_size + step_size < len(signal_chunk):
                    next_window = signal_chunk[i + step_size:i + window_size + step_size]
                    next_fft = np.abs(fft(next_window))
                    next_freqs = fftfreq(len(next_window), 1/self.sampling_rate)
                    next_target_idx = np.argmin(np.abs(next_freqs - 1000))
                    next_energy = next_fft[next_target_idx]
                    
                    if next_energy > threshold * np.max(next_fft):
                        if energy > max_energy:
                            max_energy = energy
                            best_position = i + window_size
        
        return max_energy > 0, best_position
    
    def process_audio_chunk(self, chunk):
        """处理音频数据块"""
        if not self.sync_detected:
            # 检测同步信号
            sync_found, sync_pos = self.detect_sync_signal_realtime(chunk)
            if sync_found:
                print("检测到同步信号!")
                self.sync_detected = True
                self.sync_position = sync_pos
                # 清空缓冲区，准备接收文件头
                self.audio_buffer.clear()
                return
        
        if self.sync_detected and not self.header_received:
            # 接收文件头
            self.receive_header()
            return
        
        if self.header_received and not self.file_complete:
            # 接收文件数据
            self.receive_data_chunk()
    
    def receive_header(self):
        """接收文件头"""
        # 需要累积足够的音频数据来解调文件头
        if len(self.audio_buffer) < int(self.sampling_rate * 2):
            return
        
        # 提取文件头部分（同步信号后的2秒）
        header_signal = np.array(list(self.audio_buffer))
        
        try:
            # 解调文件头
            header_bits = self.modulator.demodulate_qam(header_signal)
            
            # 解析文件头
            file_size, file_hash, protocol = self.parse_file_header(header_bits)
            
            if file_size is not None:
                print(f"文件头解析成功:")
                print(f"  文件大小: {file_size} 字节")
                print(f"  文件哈希: {file_hash}")
                print(f"  协议类型: {protocol}")
                
                self.file_size = file_size
                self.file_hash = file_hash
                self.protocol = protocol
                self.expected_bits = file_size * 8
                self.header_received = True
                
                # 清空缓冲区，准备接收数据
                self.audio_buffer.clear()
                
        except Exception as e:
            print(f"文件头解析错误: {e}")
    
    def receive_data_chunk(self):
        """接收数据块"""
        if len(self.audio_buffer) < self.samples_per_symbol * 10:  # 至少10个符号
            return
        
        try:
            # 解调数据块
            signal_array = np.array(list(self.audio_buffer))
            bits = self.modulator.demodulate_qam(signal_array)
            
            # 添加到接收缓冲区
            self.received_bits.extend(bits)
            
            # 检查是否接收完成
            if len(self.received_bits) >= self.expected_bits:
                self.file_complete = True
                print(f"文件接收完成! 接收了 {len(self.received_bits)} 比特")
            
            # 更新进度
            if self.progress_callback and self.expected_bits > 0:
                progress = min(100, len(self.received_bits) / self.expected_bits * 100)
                current_time = time.time()
                
                # 限制进度更新频率
                if current_time - self.last_progress_time > 0.1:  # 每100ms更新一次
                    self.progress_callback(progress, len(self.received_bits), self.expected_bits)
                    self.last_progress_time = current_time
            
            # 清空已处理的数据
            self.audio_buffer.clear()
            
        except Exception as e:
            print(f"数据解调错误: {e}")
    
    def parse_file_header(self, header_bits):
        """解析文件头信息"""
        try:
            # 确保有足够的比特
            if len(header_bits) < 224:  # 28字节 * 8比特
                return None, None, None
                
            # 转换为字节
            header_bytes = int(''.join(map(str, header_bits[:224])), 2).to_bytes(28, byteorder='big')
            
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
    
    def processing_worker(self):
        """数据处理工作线程"""
        while self.is_processing:
            if len(self.audio_buffer) > self.chunk_size:
                # 提取数据块进行处理
                chunk = []
                for _ in range(min(self.chunk_size, len(self.audio_buffer))):
                    chunk.append(self.audio_buffer.popleft())
                
                if chunk:
                    self.process_audio_chunk(np.array(chunk))
            elif self.sync_detected and not self.file_complete:
                # 即使缓冲区较小，也要处理数据
                if len(self.audio_buffer) > self.samples_per_symbol:
                    chunk = []
                    for _ in range(min(self.samples_per_symbol * 5, len(self.audio_buffer))):
                        chunk.append(self.audio_buffer.popleft())
                    
                    if chunk:
                        self.process_audio_chunk(np.array(chunk))
            
            time.sleep(0.005)  # 5ms间隔，提高响应速度
    
    def start_receiving(self, output_file):
        """开始接收文件"""
        if self.is_recording:
            print("已经在接收中...")
            return False
        
        print("开始实时接收文件...")
        print("等待同步信号...")
        
        # 重置状态
        self.sync_detected = False
        self.header_received = False
        self.file_complete = False
        self.received_bits = []
        self.audio_buffer.clear()
        self.start_time = time.time()
        self.last_progress_time = 0
        
        # 启动音频流
        self.is_recording = True
        self.is_processing = True
        
        try:
            # 启动音频流
            self.audio_stream = sd.InputStream(
                samplerate=self.sampling_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.chunk_size
            )
            self.audio_stream.start()
            
            # 启动处理线程
            self.processing_thread = threading.Thread(target=self.processing_worker)
            self.processing_thread.start()
            
            # 等待文件接收完成
            while self.is_recording and not self.file_complete:
                time.sleep(0.1)
                
                # 检查超时
                if not self.sync_detected and len(self.audio_buffer) > self.max_buffer_samples:
                    print("同步信号检测超时")
                    break
            
            # 停止接收
            self.stop_receiving()
            
            # 保存文件
            if self.file_complete and self.received_bits:
                return self.save_file(output_file)
            else:
                print("文件接收未完成")
                return False
                
        except Exception as e:
            print(f"接收过程中发生错误: {e}")
            self.stop_receiving()
            return False
    
    def stop_receiving(self):
        """停止接收"""
        self.is_recording = False
        self.is_processing = False
        
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)
    
    def save_file(self, output_file):
        """保存接收到的文件"""
        try:
            # 截取到正确的长度
            if len(self.received_bits) > self.expected_bits:
                self.received_bits = self.received_bits[:self.expected_bits]
            
            # 转换为字节
            byte_data = bytearray()
            for i in range(0, len(self.received_bits), 8):
                if i + 8 <= len(self.received_bits):
                    byte_bits = self.received_bits[i:i+8]
                    byte_value = int("".join(map(str, byte_bits)), 2)
                    byte_data.append(byte_value)
            
            # 保存文件
            with open(output_file, 'wb') as f:
                f.write(byte_data)
            
            print(f"文件已保存为: {output_file}")
            
            # 验证文件哈希
            if self.file_hash:
                received_hash = hashlib.md5(byte_data).hexdigest()
                if received_hash == self.file_hash:
                    print("文件完整性验证成功!")
                    return True
                else:
                    print(f"文件完整性验证失败: 期望 {self.file_hash}, 实际 {received_hash}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"保存文件时发生错误: {e}")
            return False


def progress_callback(progress, received_bits=0, expected_bits=0):
    """进度回调函数"""
    if expected_bits > 0:
        print(f"\r接收进度: {progress:.1f}% ({received_bits}/{expected_bits} 比特)", end='', flush=True)
    else:
        print(f"\r接收进度: {progress:.1f}%", end='', flush=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python realtime_qam_receiver.py <output_file>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    # 创建接收器
    receiver = RealtimeQAMReceiver()
    receiver.set_progress_callback(progress_callback)
    
    # 开始接收
    success = receiver.start_receiving(output_file)
    
    if success:
        print("\n文件接收成功!")
    else:
        print("\n文件接收失败!")
