import numpy as np
from scipy.signal import resample

class QAMModulator:
    """
    QAM调制解调器类，支持多种QAM阶数
    """
    def __init__(self, M, carrier_freq, symbol_rate, sampling_rate):
        """
        初始化QAM调制器
        
        Args:
            M: QAM的阶数 (4, 16, 64, 256)
            carrier_freq: 载波频率 (Hz)
            symbol_rate: 符号率 (symbols per second)
            sampling_rate: 采样率 (Hz)
        """
        self.M = M  # QAM的阶数
        self.k = int(np.log2(M))  # 每个符号的比特数
        self.qam_constellation = self.create_constellation()
        self.carrier_freq = carrier_freq  # 载波频率
        self.symbol_rate = symbol_rate  # 符号率
        self.sampling_rate = sampling_rate  # 采样率
        self.samples_per_symbol = int(sampling_rate / symbol_rate)

    def create_constellation(self):
        """
        生成QAM星座图
        
        Returns:
            numpy.ndarray: 星座点复数数组
        """
        if self.M == 4:
            # 4-QAM (QPSK)
            return np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        elif self.M == 16:
            # 16-QAM星座点
            return np.array([(x + 1j * y) for x in [-3, -1, 1, 3] for y in [-3, -1, 1, 3]]) / np.sqrt(10)
        elif self.M == 64:
            # 64-QAM星座点
            return np.array([(x + 1j * y) for x in [-7, -5, -3, -1, 1, 3, 5, 7] 
                           for y in [-7, -5, -3, -1, 1, 3, 5, 7]]) / np.sqrt(42)
        elif self.M == 256:
            # 256-QAM星座点
            return np.array([(x + 1j * y) for x in range(-15, 16, 2) 
                           for y in range(-15, 16, 2)]) / np.sqrt(170)
        else:
            raise ValueError(f"Unsupported QAM order: {self.M}. Supported orders: 4, 16, 64, 256")

    def modulate(self, bits):
        """
        将比特流调制为QAM符号
        
        Args:
            bits: 输入比特流 (numpy array or list)
            
        Returns:
            numpy.ndarray: QAM符号复数数组
        """
        bits = np.array(bits)
        # 确保比特数是符号长度的整数倍
        if len(bits) % self.k != 0:
            # 用零填充
            padding = self.k - (len(bits) % self.k)
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        
        # 将比特分组为符号
        symbols = [int("".join(map(str, bits[i:i+self.k])), 2) for i in range(0, len(bits), self.k)]
        # 将符号映射到QAM星座
        qam_signal = np.array([self.qam_constellation[sym] for sym in symbols])
        return qam_signal

    def demodulate(self, received_signal):
        """
        将QAM符号解调为比特流
        
        Args:
            received_signal: 接收到的QAM符号复数数组
            
        Returns:
            numpy.ndarray: 解调后的比特流
        """
        # 进行QAM解调，找到最近的星座点
        demodulated_symbols = []
        for sig in received_signal:
            # 计算与所有星座点的距离
            distances = np.abs(self.qam_constellation - sig)
            closest_idx = np.argmin(distances)
            demodulated_symbols.append(closest_idx)
        
        # 将符号转换为比特
        bits = []
        for sym in demodulated_symbols:
            bit_string = format(sym, f'0{self.k}b')
            bits.extend([int(b) for b in bit_string])
        
        return np.array(bits)

    def modulate_qam(self, bits):
        """
        将比特流调制为QAM载波信号
        
        Args:
            bits: 输入比特流
            
        Returns:
            numpy.ndarray: 调制后的实信号
        """
        # 首先调制为QAM符号
        qam_symbols = self.modulate(bits)
        
        # 上采样到采样率
        upsampled_symbols = np.repeat(qam_symbols, self.samples_per_symbol)
        
        # 生成时间轴
        t = np.arange(len(upsampled_symbols)) / self.sampling_rate
        
        # 生成载波
        carrier = np.exp(2j * np.pi * self.carrier_freq * t)
        
        # 调制：将QAM符号与载波相乘
        modulated_complex = upsampled_symbols * carrier
        
        # 返回实部（I路信号）
        return np.real(modulated_complex)

    def demodulate_qam(self, signal):
        """
        将QAM载波信号解调为比特流
        
        Args:
            signal: 接收到的实信号
            
        Returns:
            numpy.ndarray: 解调后的比特流
        """
        # 生成时间轴
        t = np.arange(len(signal)) / self.sampling_rate
        
        # 生成载波进行解调
        carrier = np.exp(-2j * np.pi * self.carrier_freq * t)
        
        # 解调：与载波相乘
        demodulated_complex = signal * carrier
        
        # 简单的低通滤波（移动平均）
        window_size = self.samples_per_symbol
        if len(demodulated_complex) >= window_size:
            filtered_signal = np.convolve(demodulated_complex, np.ones(window_size)/window_size, mode='valid')
        else:
            filtered_signal = demodulated_complex
        
        # 抽取到符号率（每samples_per_symbol个样本取一个）
        downsampled_symbols = filtered_signal[::self.samples_per_symbol]
        
        # 星座图匹配
        decoded_bits = []
        for sym in downsampled_symbols:
            # 找到与接收到的符号最接近的星座点
            distances = np.abs(self.qam_constellation - sym)
            closest_idx = np.argmin(distances)
            bit_string = format(closest_idx, f'0{self.k}b')
            decoded_bits.extend([int(b) for b in bit_string])
        
        return np.array(decoded_bits)


