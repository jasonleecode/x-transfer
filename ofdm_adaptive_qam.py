import numpy as np
import sounddevice as sd
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import resample, butter, filtfilt
import threading
import time
from enum import Enum
from typing import List, Tuple, Optional
import struct

class ModulationType(Enum):
    """调制类型枚举"""
    QPSK = 4
    QAM16 = 16
    QAM64 = 64
    QAM256 = 256

class ChannelState(Enum):
    """信道状态枚举"""
    EXCELLENT = "excellent"  # 优秀
    GOOD = "good"           # 良好
    FAIR = "fair"           # 一般
    POOR = "poor"           # 较差

class OFDMAdaptiveQAM:
    """
    OFDM + 自适应QAM调制解调器
    
    特性：
    - 多子载波并行传输
    - 自适应调制编码
    - 信道估计和均衡
    - 导频符号插入
    - 循环前缀
    - 功率控制
    """
    
    def __init__(self, 
                 sampling_rate=44100,
                 num_subcarriers=64,
                 num_pilot_carriers=8,
                 cp_ratio=0.25,
                 base_freq=1000,
                 freq_spacing=100):
        """
        初始化OFDM自适应QAM调制器
        
        Args:
            sampling_rate: 采样率 (Hz)
            num_subcarriers: 子载波数量
            num_pilot_carriers: 导频载波数量
            cp_ratio: 循环前缀比例
            base_freq: 基础频率 (Hz)
            freq_spacing: 子载波频率间隔 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.num_subcarriers = num_subcarriers
        self.num_pilot_carriers = num_pilot_carriers
        self.num_data_carriers = num_subcarriers - num_pilot_carriers
        self.cp_ratio = cp_ratio
        self.base_freq = base_freq
        self.freq_spacing = freq_spacing
        
        # 计算OFDM参数
        self.symbol_duration = 1.0 / freq_spacing  # 符号持续时间
        self.cp_duration = self.symbol_duration * cp_ratio  # 循环前缀持续时间
        self.total_symbol_duration = self.symbol_duration + self.cp_duration
        self.samples_per_symbol = int(self.sampling_rate * self.symbol_duration)
        self.cp_samples = int(self.sampling_rate * self.cp_duration)
        self.total_samples_per_symbol = self.samples_per_symbol + self.cp_samples
        
        # 子载波频率
        self.subcarrier_freqs = np.array([
            base_freq + i * freq_spacing 
            for i in range(num_subcarriers)
        ])
        
        # 导频载波位置（均匀分布）
        self.pilot_positions = np.linspace(0, num_subcarriers-1, num_pilot_carriers, dtype=int)
        self.data_positions = np.setdiff1d(range(num_subcarriers), self.pilot_positions)
        
        # 导频符号（固定序列）
        self.pilot_symbols = np.array([1+1j, -1+1j, -1-1j, 1-1j] * (num_pilot_carriers // 4 + 1))[:num_pilot_carriers]
        
        # 星座图
        self.constellations = self._create_constellations()
        
        # 信道状态
        self.channel_state = ChannelState.GOOD
        self.snr_estimate = 20.0  # 信噪比估计 (dB)
        self.channel_response = None
        
        # 自适应参数
        self.adaptive_modulation = True
        self.min_modulation = ModulationType.QPSK
        self.max_modulation = ModulationType.QAM256
        
        # 功率控制
        self.power_control = True
        self.max_power = 0.5
        self.min_power = 0.1
        
    def _create_constellations(self):
        """创建各种调制方式的星座图"""
        constellations = {}
        
        # QPSK
        constellations[ModulationType.QPSK] = np.array([
            1+1j, -1+1j, -1-1j, 1-1j
        ]) / np.sqrt(2)
        
        # 16-QAM
        constellations[ModulationType.QAM16] = np.array([
            (x + 1j * y) for x in [-3, -1, 1, 3] for y in [-3, -1, 1, 3]
        ]) / np.sqrt(10)
        
        # 64-QAM
        constellations[ModulationType.QAM64] = np.array([
            (x + 1j * y) for x in [-7, -5, -3, -1, 1, 3, 5, 7] 
            for y in [-7, -5, -3, -1, 1, 3, 5, 7]
        ]) / np.sqrt(42)
        
        # 256-QAM
        constellations[ModulationType.QAM256] = np.array([
            (x + 1j * y) for x in range(-15, 16, 2) 
            for y in range(-15, 16, 2)
        ]) / np.sqrt(170)
        
        return constellations
    
    def _estimate_channel_state(self, received_pilots, transmitted_pilots):
        """估计信道状态"""
        try:
            # 计算信道响应
            channel_response = received_pilots / transmitted_pilots
            
            # 计算信噪比
            signal_power = np.mean(np.abs(received_pilots) ** 2)
            noise_power = np.var(received_pilots - transmitted_pilots * channel_response)
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            self.snr_estimate = snr_db
            self.channel_response = channel_response
            
            # 根据SNR确定信道状态
            if snr_db > 25:
                self.channel_state = ChannelState.EXCELLENT
            elif snr_db > 15:
                self.channel_state = ChannelState.GOOD
            elif snr_db > 8:
                self.channel_state = ChannelState.FAIR
            else:
                self.channel_state = ChannelState.POOR
                
            return self.channel_state, snr_db
            
        except Exception as e:
            print(f"信道估计错误: {e}")
            return ChannelState.POOR, 0.0
    
    def _select_modulation(self, snr_db):
        """根据SNR选择调制方式"""
        if not self.adaptive_modulation:
            return ModulationType.QAM16
        
        if snr_db > 25:
            return ModulationType.QAM256
        elif snr_db > 15:
            return ModulationType.QAM64
        elif snr_db > 8:
            return ModulationType.QAM16
        else:
            return ModulationType.QPSK
    
    def _calculate_power_allocation(self, modulation_type):
        """计算功率分配"""
        if not self.power_control:
            return np.ones(self.num_data_carriers) * self.max_power
        
        # 根据调制方式调整功率
        power_levels = {
            ModulationType.QPSK: 1.0,
            ModulationType.QAM16: 0.8,
            ModulationType.QAM64: 0.6,
            ModulationType.QAM256: 0.4
        }
        
        base_power = power_levels[modulation_type] * self.max_power
        return np.ones(self.num_data_carriers) * base_power
    
    def _modulate_symbols(self, bits, modulation_type):
        """调制符号"""
        constellation = self.constellations[modulation_type]
        k = int(np.log2(len(constellation)))
        
        # 确保比特数是符号长度的整数倍
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
    
    def _demodulate_symbols(self, received_symbols, modulation_type):
        """解调符号"""
        constellation = self.constellations[modulation_type]
        
        demodulated_bits = []
        for symbol in received_symbols:
            # 找到最近的星座点
            distances = np.abs(constellation - symbol)
            closest_idx = np.argmin(distances)
            
            # 转换为比特
            k = int(np.log2(len(constellation)))
            bit_string = format(closest_idx, f'0{k}b')
            demodulated_bits.extend([int(b) for b in bit_string])
        
        return np.array(demodulated_bits)
    
    def _apply_channel_estimation(self, received_symbols):
        """应用信道估计和均衡"""
        if self.channel_response is None:
            return received_symbols
        
        # 简单的零强制均衡
        equalized_symbols = received_symbols / self.channel_response
        return equalized_symbols
    
    def modulate_ofdm(self, bits, modulation_type=None):
        """
        OFDM调制
        
        Args:
            bits: 输入比特流
            modulation_type: 调制类型，None表示自适应选择
            
        Returns:
            numpy.ndarray: 调制后的OFDM信号
        """
        if modulation_type is None:
            modulation_type = self._select_modulation(self.snr_estimate)
        
        # 调制数据符号
        data_symbols = self._modulate_symbols(bits, modulation_type)
        
        # 确保有足够的数据符号
        symbols_per_frame = self.num_data_carriers
        num_frames = (len(data_symbols) + symbols_per_frame - 1) // symbols_per_frame
        
        # 填充数据
        padded_data = np.concatenate([
            data_symbols, 
            np.zeros(num_frames * symbols_per_frame - len(data_symbols), dtype=complex)
        ])
        
        # 重塑为帧
        data_frames = padded_data.reshape(num_frames, symbols_per_frame)
        
        # 生成OFDM信号
        ofdm_signal = []
        
        for frame in data_frames:
            # 创建OFDM符号
            ofdm_symbol = np.zeros(self.num_subcarriers, dtype=complex)
            
            # 插入数据符号
            ofdm_symbol[self.data_positions] = frame
            
            # 插入导频符号
            ofdm_symbol[self.pilot_positions] = self.pilot_symbols
            
            # IFFT变换
            time_symbol = ifft(ofdm_symbol)
            
            # 添加循环前缀
            cp = time_symbol[-self.cp_samples:]
            symbol_with_cp = np.concatenate([cp, time_symbol])
            
            ofdm_signal.extend(symbol_with_cp)
        
        return np.array(ofdm_signal)
    
    def demodulate_ofdm(self, signal):
        """
        OFDM解调
        
        Args:
            signal: 接收到的OFDM信号
            
        Returns:
            tuple: (解调后的比特流, 使用的调制类型)
        """
        # 计算帧数
        num_frames = len(signal) // self.total_samples_per_symbol
        
        if num_frames == 0:
            return np.array([]), ModulationType.QPSK
        
        # 重塑信号
        signal_frames = signal[:num_frames * self.total_samples_per_symbol].reshape(
            num_frames, self.total_samples_per_symbol
        )
        
        demodulated_bits = []
        used_modulation = ModulationType.QPSK
        
        for frame in signal_frames:
            # 移除循环前缀
            symbol = frame[self.cp_samples:]
            
            # FFT变换
            freq_symbol = fft(symbol)
            
            # 提取导频符号进行信道估计
            received_pilots = freq_symbol[self.pilot_positions]
            transmitted_pilots = self.pilot_symbols
            
            # 信道估计
            channel_state, snr_db = self._estimate_channel_state(received_pilots, transmitted_pilots)
            
            # 选择调制方式
            modulation_type = self._select_modulation(snr_db)
            used_modulation = modulation_type
            
            # 提取数据符号
            received_data = freq_symbol[self.data_positions]
            
            # 信道均衡
            equalized_data = self._apply_channel_estimation(received_data)
            
            # 解调符号
            frame_bits = self._demodulate_symbols(equalized_data, modulation_type)
            demodulated_bits.extend(frame_bits)
        
        return np.array(demodulated_bits), used_modulation
    
    def modulate_with_carrier(self, bits, modulation_type=None):
        """
        带载波的OFDM调制
        
        Args:
            bits: 输入比特流
            modulation_type: 调制类型
            
        Returns:
            numpy.ndarray: 调制后的实信号
        """
        # OFDM调制
        ofdm_signal = self.modulate_ofdm(bits, modulation_type)
        
        # 生成时间轴
        t = np.arange(len(ofdm_signal)) / self.sampling_rate
        
        # 生成载波
        carrier = np.exp(2j * np.pi * self.base_freq * t)
        
        # 调制载波
        modulated_signal = ofdm_signal * carrier
        
        # 返回实部
        return np.real(modulated_signal)
    
    def demodulate_with_carrier(self, signal):
        """
        带载波的OFDM解调
        
        Args:
            signal: 接收到的实信号
            
        Returns:
            tuple: (解调后的比特流, 使用的调制类型)
        """
        # 生成时间轴
        t = np.arange(len(signal)) / self.sampling_rate
        
        # 生成载波进行解调
        carrier = np.exp(-2j * np.pi * self.base_freq * t)
        
        # 解调载波
        demodulated_signal = signal * carrier
        
        # 低通滤波
        nyquist = self.sampling_rate / 2
        cutoff = self.base_freq + self.num_subcarriers * self.freq_spacing / 2
        normal_cutoff = cutoff / nyquist
        
        if normal_cutoff < 1.0:
            b, a = butter(5, normal_cutoff, btype='low', analog=False)
            demodulated_signal = filtfilt(b, a, demodulated_signal)
        
        # OFDM解调
        return self.demodulate_ofdm(demodulated_signal)
    
    def get_spectral_efficiency(self, modulation_type):
        """获取频谱效率 (bits/symbol)"""
        return int(np.log2(modulation_type.value))
    
    def get_estimated_capacity(self):
        """获取估计的信道容量 (bits/symbol)"""
        if self.channel_state == ChannelState.EXCELLENT:
            return self.get_spectral_efficiency(ModulationType.QAM256)
        elif self.channel_state == ChannelState.GOOD:
            return self.get_spectral_efficiency(ModulationType.QAM64)
        elif self.channel_state == ChannelState.FAIR:
            return self.get_spectral_efficiency(ModulationType.QAM16)
        else:
            return self.get_spectral_efficiency(ModulationType.QPSK)
    
    def get_channel_info(self):
        """获取信道信息"""
        return {
            'channel_state': self.channel_state.value,
            'snr_estimate': self.snr_estimate,
            'estimated_capacity': self.get_estimated_capacity(),
            'num_subcarriers': self.num_subcarriers,
            'num_data_carriers': self.num_data_carriers,
            'freq_spacing': self.freq_spacing
        }
    
    def set_adaptive_modulation(self, enabled):
        """设置自适应调制"""
        self.adaptive_modulation = enabled
    
    def set_power_control(self, enabled):
        """设置功率控制"""
        self.power_control = enabled
    
    def set_modulation_range(self, min_mod, max_mod):
        """设置调制范围"""
        self.min_modulation = min_mod
        self.max_modulation = max_mod


class OFDMAudioTransceiver:
    """
    OFDM音频收发器 - 封装音频接口
    """
    
    def __init__(self, ofdm_modulator, sampling_rate=44100):
        self.ofdm_modulator = ofdm_modulator
        self.sampling_rate = sampling_rate
        self.is_transmitting = False
        self.is_receiving = False
        
    def send_signal(self, signal):
        """发送信号"""
        try:
            sd.play(signal, self.sampling_rate)
            sd.wait()
            return True
        except Exception as e:
            print(f"发送信号错误: {e}")
            return False
    
    def receive_signal(self, duration):
        """接收信号"""
        try:
            signal = sd.rec(int(duration * self.sampling_rate), 
                           samplerate=self.sampling_rate, channels=1)
            sd.wait()
            return signal[:, 0]
        except Exception as e:
            print(f"接收信号错误: {e}")
            return None
    
    def transmit_bits(self, bits, modulation_type=None):
        """传输比特流"""
        # 调制
        signal = self.ofdm_modulator.modulate_with_carrier(bits, modulation_type)
        
        # 发送
        return self.send_signal(signal)
    
    def receive_bits(self, duration, expected_modulation=None):
        """接收比特流"""
        # 接收
        signal = self.receive_signal(duration)
        if signal is None:
            return None, None
        
        # 解调
        bits, used_modulation = self.ofdm_modulator.demodulate_with_carrier(signal)
        
        return bits, used_modulation


# 测试和示例代码
if __name__ == "__main__":
    # 创建OFDM调制器
    ofdm_mod = OFDMAdaptiveQAM(
        sampling_rate=44100,
        num_subcarriers=32,  # 减少子载波数量便于测试
        num_pilot_carriers=4,
        base_freq=2000,
        freq_spacing=200
    )
    
    # 创建音频收发器
    transceiver = OFDMAudioTransceiver(ofdm_mod)
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 1000)
    print(f"原始比特数: {len(test_bits)}")
    
    # 显示信道信息
    print("信道信息:")
    info = ofdm_mod.get_channel_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试调制
    print("\n测试OFDM调制...")
    signal = ofdm_mod.modulate_with_carrier(test_bits)
    print(f"调制信号长度: {len(signal)} 样本")
    print(f"信号持续时间: {len(signal) / ofdm_mod.sampling_rate:.2f} 秒")
    
    # 测试解调
    print("\n测试OFDM解调...")
    demodulated_bits, used_mod = ofdm_mod.demodulate_with_carrier(signal)
    print(f"解调比特数: {len(demodulated_bits)}")
    print(f"使用调制: {used_mod.name}")
    
    # 计算误码率
    if len(demodulated_bits) >= len(test_bits):
        error_rate = np.mean(demodulated_bits[:len(test_bits)] != test_bits)
        print(f"误码率: {error_rate:.4f}")
    
    print("\nOFDM自适应QAM模块测试完成!")
