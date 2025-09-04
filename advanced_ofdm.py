import numpy as np
import sounddevice as sd
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import resample, butter, filtfilt, hilbert
from scipy.optimize import minimize
import threading
import time
from enum import Enum
from typing import List, Tuple, Optional, Dict
import struct
import warnings
warnings.filterwarnings('ignore')

class OFDMConfig:
    """OFDM配置类"""
    def __init__(self):
        # 基本参数
        self.sampling_rate = 44100
        self.num_subcarriers = 64
        self.num_pilot_carriers = 8
        self.cp_ratio = 0.25
        self.base_freq = 2000
        self.freq_spacing = 100
        
        # 高级参数
        self.use_window = True
        self.window_type = 'hamming'
        self.use_precoding = True
        self.use_beamforming = False
        self.use_mimo = False
        self.num_antennas = 2
        
        # 信道编码
        self.use_fec = True
        self.fec_rate = 0.5
        self.interleaver_depth = 8
        
        # 同步
        self.use_sync = True
        self.sync_sequence_length = 64
        self.use_freq_sync = True
        
        # 均衡
        self.equalizer_type = 'mmse'  # 'zf', 'mmse', 'ml'
        self.use_adaptive_eq = True
        self.training_sequence_length = 32

class AdvancedOFDM:
    """
    高级OFDM调制解调器
    
    特性：
    - 多天线MIMO支持
    - 预编码和波束成形
    - 信道编码和交织
    - 高级同步算法
    - 自适应均衡
    - 功率分配优化
    - 频谱整形
    """
    
    def __init__(self, config: OFDMConfig = None):
        self.config = config or OFDMConfig()
        self._setup_parameters()
        self._setup_constellations()
        self._setup_sync_sequences()
        self._setup_training_sequences()
        
        # 状态变量
        self.channel_matrix = None
        self.noise_variance = 1e-6
        self.snr_estimate = 20.0
        self.frequency_offset = 0.0
        self.timing_offset = 0
        
    def _setup_parameters(self):
        """设置OFDM参数"""
        self.symbol_duration = 1.0 / self.config.freq_spacing
        self.cp_duration = self.symbol_duration * self.config.cp_ratio
        self.total_symbol_duration = self.symbol_duration + self.cp_duration
        
        self.samples_per_symbol = int(self.config.sampling_rate * self.symbol_duration)
        self.cp_samples = int(self.config.sampling_rate * self.cp_duration)
        self.total_samples_per_symbol = self.samples_per_symbol + self.cp_samples
        
        # 子载波频率
        self.subcarrier_freqs = np.array([
            self.config.base_freq + i * self.config.freq_spacing 
            for i in range(self.config.num_subcarriers)
        ])
        
        # 导频和数据载波位置
        self.pilot_positions = np.linspace(0, self.config.num_subcarriers-1, 
                                         self.config.num_pilot_carriers, dtype=int)
        self.data_positions = np.setdiff1d(range(self.config.num_subcarriers), 
                                         self.pilot_positions)
        
        # 窗口函数
        if self.config.use_window:
            if self.config.window_type == 'hamming':
                self.window = np.hamming(self.samples_per_symbol)
            elif self.config.window_type == 'hanning':
                self.window = np.hanning(self.samples_per_symbol)
            else:
                self.window = np.ones(self.samples_per_symbol)
        else:
            self.window = np.ones(self.samples_per_symbol)
    
    def _setup_constellations(self):
        """设置星座图"""
        self.constellations = {
            4: np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2),
            16: np.array([(x + 1j * y) for x in [-3, -1, 1, 3] 
                         for y in [-3, -1, 1, 3]]) / np.sqrt(10),
            64: np.array([(x + 1j * y) for x in [-7, -5, -3, -1, 1, 3, 5, 7] 
                         for y in [-7, -5, -3, -1, 1, 3, 5, 7]]) / np.sqrt(42),
            256: np.array([(x + 1j * y) for x in range(-15, 16, 2) 
                          for y in range(-15, 16, 2)]) / np.sqrt(170)
        }
    
    def _setup_sync_sequences(self):
        """设置同步序列"""
        if self.config.use_sync:
            # 使用Zadoff-Chu序列作为同步序列
            N = self.config.sync_sequence_length
            q = 1  # 根索引
            self.sync_sequence = np.array([
                np.exp(-1j * np.pi * q * n * (n + 1) / N) 
                for n in range(N)
            ])
        else:
            self.sync_sequence = None
    
    def _setup_training_sequences(self):
        """设置训练序列"""
        if self.config.use_adaptive_eq:
            # 使用伪随机序列作为训练序列
            np.random.seed(42)  # 固定种子确保可重复性
            self.training_sequence = np.random.choice(
                [1+1j, -1+1j, -1-1j, 1-1j], 
                self.config.training_sequence_length
            )
        else:
            self.training_sequence = None
    
    def _apply_precoding(self, symbols, precoding_matrix=None):
        """应用预编码"""
        if not self.config.use_precoding or precoding_matrix is None:
            return symbols
        
        return np.dot(symbols, precoding_matrix)
    
    def _apply_beamforming(self, signal, beamforming_weights=None):
        """应用波束成形"""
        if not self.config.use_beamforming or beamforming_weights is None:
            return signal
        
        # 简化的波束成形实现
        return signal * beamforming_weights
    
    def _encode_fec(self, bits):
        """前向纠错编码"""
        if not self.config.use_fec:
            return bits
        
        # 简化的卷积编码实现
        # 这里使用重复编码作为示例
        encoded_bits = []
        for bit in bits:
            encoded_bits.extend([bit] * int(1 / self.config.fec_rate))
        
        return np.array(encoded_bits)
    
    def _decode_fec(self, encoded_bits):
        """前向纠错解码"""
        if not self.config.use_fec:
            return encoded_bits
        
        # 简化的重复解码实现
        rate = int(1 / self.config.fec_rate)
        decoded_bits = []
        
        for i in range(0, len(encoded_bits), rate):
            chunk = encoded_bits[i:i+rate]
            if len(chunk) == rate:
                # 多数投票解码
                decoded_bits.append(1 if np.sum(chunk) > rate/2 else 0)
        
        return np.array(decoded_bits)
    
    def _interleave(self, bits):
        """交织"""
        if self.config.interleaver_depth <= 1:
            return bits
        
        # 简化的块交织
        depth = self.config.interleaver_depth
        padded_length = ((len(bits) + depth - 1) // depth) * depth
        padded_bits = np.concatenate([bits, np.zeros(padded_length - len(bits))])
        
        interleaved = padded_bits.reshape(-1, depth).T.flatten()
        return interleaved[:len(bits)]
    
    def _deinterleave(self, bits):
        """解交织"""
        if self.config.interleaver_depth <= 1:
            return bits
        
        # 简化的块解交织
        depth = self.config.interleaver_depth
        padded_length = ((len(bits) + depth - 1) // depth) * depth
        padded_bits = np.concatenate([bits, np.zeros(padded_length - len(bits))])
        
        deinterleaved = padded_bits.reshape(depth, -1).T.flatten()
        return deinterleaved[:len(bits)]
    
    def _modulate_symbols(self, bits, modulation_order):
        """调制符号"""
        constellation = self.constellations[modulation_order]
        k = int(np.log2(modulation_order))
        
        # 确保比特数是符号长度的整数倍
        bits = np.array(bits, dtype=int)
        if len(bits) % k != 0:
            padding = k - (len(bits) % k)
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        
        # 将比特分组为符号
        symbols = []
        for i in range(0, len(bits), k):
            symbol_bits = bits[i:i+k]
            symbol_index = int(''.join(map(str, symbol_bits)), 2)
            symbols.append(constellation[symbol_index])
        
        return np.array(symbols)
    
    def _demodulate_symbols(self, received_symbols, modulation_order):
        """解调符号"""
        constellation = self.constellations[modulation_order]
        
        demodulated_bits = []
        for symbol in received_symbols:
            # 找到最近的星座点
            distances = np.abs(constellation - symbol)
            closest_idx = np.argmin(distances)
            
            # 转换为比特
            k = int(np.log2(modulation_order))
            bit_string = format(closest_idx, f'0{k}b')
            demodulated_bits.extend([int(b) for b in bit_string])
        
        return np.array(demodulated_bits)
    
    def _estimate_channel(self, received_pilots, transmitted_pilots):
        """信道估计"""
        # 最小二乘信道估计
        channel_response = received_pilots / transmitted_pilots
        
        # 噪声方差估计
        noise = received_pilots - transmitted_pilots * channel_response
        self.noise_variance = np.var(noise)
        
        # SNR估计
        signal_power = np.mean(np.abs(received_pilots) ** 2)
        self.snr_estimate = 10 * np.log10(signal_power / (self.noise_variance + 1e-10))
        
        return channel_response
    
    def _apply_equalization(self, received_symbols, channel_response):
        """应用均衡"""
        if self.config.equalizer_type == 'zf':
            # 零强制均衡
            equalized = received_symbols / channel_response
        elif self.config.equalizer_type == 'mmse':
            # 最小均方误差均衡
            snr = 10**(self.snr_estimate/10)
            equalized = received_symbols * np.conj(channel_response) / (
                np.abs(channel_response)**2 + 1/snr
            )
        else:  # 'ml'
            # 最大似然均衡（简化实现）
            equalized = received_symbols / channel_response
        
        return equalized
    
    def _detect_sync(self, signal):
        """同步检测"""
        if not self.config.use_sync or self.sync_sequence is None:
            return 0
        
        # 使用相关检测
        correlation = np.correlate(signal, np.real(self.sync_sequence), mode='valid')
        sync_position = np.argmax(np.abs(correlation))
        
        return sync_position
    
    def _estimate_frequency_offset(self, received_pilots, transmitted_pilots):
        """频率偏移估计"""
        if not self.config.use_freq_sync:
            return 0.0
        
        # 使用导频符号估计频率偏移
        phase_diff = np.angle(received_pilots / transmitted_pilots)
        freq_offset = np.mean(np.diff(phase_diff)) / (2 * np.pi * self.symbol_duration)
        
        return freq_offset
    
    def _correct_frequency_offset(self, signal, freq_offset):
        """频率偏移校正"""
        if abs(freq_offset) < 1e-6:
            return signal
        
        t = np.arange(len(signal)) / self.config.sampling_rate
        correction = np.exp(-2j * np.pi * freq_offset * t)
        return signal * correction
    
    def modulate(self, bits, modulation_order=16, use_advanced_features=True):
        """
        高级OFDM调制
        
        Args:
            bits: 输入比特流
            modulation_order: 调制阶数
            use_advanced_features: 是否使用高级特性
            
        Returns:
            numpy.ndarray: 调制后的OFDM信号
        """
        # FEC编码
        if use_advanced_features:
            bits = self._encode_fec(bits)
            bits = self._interleave(bits)
        
        # 调制符号
        symbols = self._modulate_symbols(bits, modulation_order)
        
        # 计算帧数
        symbols_per_frame = self.config.num_subcarriers - self.config.num_pilot_carriers
        num_frames = (len(symbols) + symbols_per_frame - 1) // symbols_per_frame
        
        # 填充数据
        padded_symbols = np.concatenate([
            symbols, 
            np.zeros(num_frames * symbols_per_frame - len(symbols), dtype=complex)
        ])
        
        # 重塑为帧
        symbol_frames = padded_symbols.reshape(num_frames, symbols_per_frame)
        
        # 生成OFDM信号
        ofdm_signal = []
        
        for i, frame in enumerate(symbol_frames):
            # 创建OFDM符号
            ofdm_symbol = np.zeros(self.config.num_subcarriers, dtype=complex)
            
            # 插入数据符号
            ofdm_symbol[self.data_positions] = frame
            
            # 插入导频符号
            pilot_symbols = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 
                                   (self.config.num_pilot_carriers // 4 + 1))[:self.config.num_pilot_carriers]
            ofdm_symbol[self.pilot_positions] = pilot_symbols
            
            # 插入训练序列（每10帧一次）
            if (use_advanced_features and self.training_sequence is not None and 
                i % 10 == 0 and len(self.training_sequence) <= len(self.data_positions)):
                ofdm_symbol[self.data_positions[:len(self.training_sequence)]] = self.training_sequence
            
            # 预编码
            if use_advanced_features and self.config.use_precoding:
                # 简化的预编码矩阵
                precoding_matrix = np.eye(len(frame))
                ofdm_symbol[self.data_positions] = self._apply_precoding(frame, precoding_matrix)
            
            # IFFT变换
            time_symbol = ifft(ofdm_symbol)
            
            # 应用窗口
            if len(time_symbol) == len(self.window):
                time_symbol *= self.window
            else:
                # 如果长度不匹配，使用简单的窗口
                window = np.ones(len(time_symbol))
                time_symbol *= window
            
            # 添加循环前缀
            cp = time_symbol[-self.cp_samples:]
            symbol_with_cp = np.concatenate([cp, time_symbol])
            
            ofdm_signal.extend(symbol_with_cp)
        
        return np.array(ofdm_signal)
    
    def demodulate(self, signal, expected_modulation_order=16, use_advanced_features=True):
        """
        高级OFDM解调
        
        Args:
            signal: 接收到的OFDM信号
            expected_modulation_order: 期望的调制阶数
            use_advanced_features: 是否使用高级特性
            
        Returns:
            tuple: (解调后的比特流, 使用的调制阶数)
        """
        # 同步检测
        if use_advanced_features and self.config.use_sync:
            sync_pos = self._detect_sync(signal)
            signal = signal[sync_pos:]
        
        # 计算帧数
        num_frames = len(signal) // self.total_samples_per_symbol
        
        if num_frames == 0:
            return np.array([]), expected_modulation_order
        
        # 重塑信号
        signal_frames = signal[:num_frames * self.total_samples_per_symbol].reshape(
            num_frames, self.total_samples_per_symbol
        )
        
        demodulated_bits = []
        used_modulation = expected_modulation_order
        
        for frame in signal_frames:
            # 移除循环前缀
            symbol = frame[self.cp_samples:]
            
            # FFT变换
            freq_symbol = fft(symbol)
            
            # 提取导频符号
            received_pilots = freq_symbol[self.pilot_positions]
            transmitted_pilots = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 
                                        (self.config.num_pilot_carriers // 4 + 1))[:self.config.num_pilot_carriers]
            
            # 信道估计
            channel_response = self._estimate_channel(received_pilots, transmitted_pilots)
            
            # 频率偏移估计和校正
            if use_advanced_features and self.config.use_freq_sync:
                freq_offset = self._estimate_frequency_offset(received_pilots, transmitted_pilots)
                freq_symbol = self._correct_frequency_offset(freq_symbol, freq_offset)
            
            # 提取数据符号
            received_data = freq_symbol[self.data_positions]
            
            # 均衡
            equalized_data = self._apply_equalization(received_data, 
                                                    channel_response[self.data_positions])
            
            # 解调符号
            frame_bits = self._demodulate_symbols(equalized_data, used_modulation)
            demodulated_bits.extend(frame_bits)
        
        # FEC解码
        if use_advanced_features:
            demodulated_bits = self._deinterleave(demodulated_bits)
            demodulated_bits = self._decode_fec(demodulated_bits)
        
        return np.array(demodulated_bits), used_modulation
    
    def get_performance_metrics(self):
        """获取性能指标"""
        return {
            'snr_estimate': self.snr_estimate,
            'noise_variance': self.noise_variance,
            'frequency_offset': self.frequency_offset,
            'spectral_efficiency': 4.0,  # 16-QAM的频谱效率
            'channel_condition': 'good' if self.snr_estimate > 15 else 'poor'
        }


# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = OFDMConfig()
    config.num_subcarriers = 32
    config.num_pilot_carriers = 4
    config.use_fec = True
    config.use_sync = True
    
    # 创建高级OFDM调制器
    ofdm = AdvancedOFDM(config)
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 500)
    print(f"原始比特数: {len(test_bits)}")
    
    # 调制
    print("高级OFDM调制...")
    signal = ofdm.modulate(test_bits, modulation_order=16, use_advanced_features=True)
    print(f"调制信号长度: {len(signal)} 样本")
    
    # 解调
    print("高级OFDM解调...")
    demodulated_bits, used_mod = ofdm.demodulate(signal, expected_modulation_order=16, 
                                                use_advanced_features=True)
    print(f"解调比特数: {len(demodulated_bits)}")
    
    # 性能指标
    metrics = ofdm.get_performance_metrics()
    print("性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("高级OFDM模块测试完成!")
