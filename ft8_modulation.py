import numpy as np
import sounddevice as sd
from scipy.signal import chirp
import hashlib

class FT8Modulator:
    """
    FT8调制器 - 基于WSJT-X的FT8协议
    FT8是一种数字通信模式，用于业余无线电
    """
    def __init__(self, sampling_rate=12000, symbol_rate=6.25):
        """
        初始化FT8调制器
        
        Args:
            sampling_rate: 采样率（Hz），FT8通常使用12kHz
            symbol_rate: 符号率（Hz），FT8使用6.25 baud
        """
        self.sampling_rate = sampling_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(sampling_rate / symbol_rate)
        
        # FT8参数
        self.tones = 8  # 8个音调
        self.tone_spacing = 6.25  # 音调间隔（Hz）
        self.base_freq = 1000  # 基础频率（Hz）
        self.symbol_duration = 1.0 / symbol_rate  # 符号持续时间
        
        # 生成音调频率
        self.tone_frequencies = [self.base_freq + i * self.tone_spacing 
                                for i in range(self.tones)]

    def encode_message(self, message, callsign1="CQ", callsign2="", grid=""):
        """
        编码消息为FT8格式
        
        Args:
            message: 要发送的消息
            callsign1: 第一个呼号
            callsign2: 第二个呼号（可选）
            grid: 网格定位（可选）
            
        Returns:
            list: 编码后的符号序列
        """
        # 简化的FT8编码（实际FT8编码更复杂）
        # 这里实现一个基本版本
        
        # 将消息转换为数字序列
        symbols = []
        for char in message.upper():
            if char.isalpha():
                symbols.append(ord(char) - ord('A'))
            elif char.isdigit():
                symbols.append(26 + int(char))
            elif char == ' ':
                symbols.append(36)
            else:
                symbols.append(37)  # 其他字符
        
        # 添加呼号信息
        if callsign1:
            for char in callsign1.upper():
                if char.isalpha():
                    symbols.append(ord(char) - ord('A'))
                elif char.isdigit():
                    symbols.append(26 + int(char))
        
        if callsign2:
            symbols.append(38)  # 分隔符
            for char in callsign2.upper():
                if char.isalpha():
                    symbols.append(ord(char) - ord('A'))
                elif char.isdigit():
                    symbols.append(26 + int(char))
        
        if grid:
            symbols.append(39)  # 网格分隔符
            for char in grid.upper():
                if char.isalpha():
                    symbols.append(ord(char) - ord('A'))
                elif char.isdigit():
                    symbols.append(26 + int(char))
        
        return symbols

    def generate_ft8_signal(self, symbols, duration=12.0):
        """
        生成FT8信号
        
        Args:
            symbols: 符号序列
            duration: 信号持续时间（秒）
            
        Returns:
            numpy.ndarray: FT8音频信号
        """
        total_samples = int(duration * self.sampling_rate)
        signal = np.zeros(total_samples)
        
        # 计算每个符号的样本数
        samples_per_symbol = int(self.sampling_rate / self.symbol_rate)
        
        # 生成每个符号
        for i, symbol in enumerate(symbols):
            start_sample = i * samples_per_symbol
            end_sample = min(start_sample + samples_per_symbol, total_samples)
            
            if start_sample < total_samples:
                # 选择音调频率
                tone_idx = symbol % self.tones
                frequency = self.tone_frequencies[tone_idx]
                
                # 生成符号信号
                symbol_samples = end_sample - start_sample
                t = np.linspace(0, self.symbol_duration, symbol_samples, endpoint=False)
                
                # 使用正弦波生成音调
                symbol_signal = 0.3 * np.sin(2 * np.pi * frequency * t)
                
                # 添加平滑的上升和下降沿
                fade_samples = int(0.01 * self.sampling_rate)  # 10ms淡入淡出
                if len(symbol_signal) > 2 * fade_samples:
                    # 淡入
                    symbol_signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    # 淡出
                    symbol_signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                signal[start_sample:end_sample] = symbol_signal
        
        return signal

    def add_sync_tones(self, signal):
        """
        添加同步音调（简化版本）
        
        Args:
            signal: 输入信号
            
        Returns:
            numpy.ndarray: 添加同步音调后的信号
        """
        # 在信号开始和结束添加同步音调
        sync_duration = 0.5  # 同步音调持续时间
        sync_samples = int(sync_duration * self.sampling_rate)
        
        # 开始同步音调
        t_start = np.linspace(0, sync_duration, sync_samples, endpoint=False)
        sync_start = 0.3 * np.sin(2 * np.pi * self.base_freq * t_start)
        signal[:sync_samples] = sync_start
        
        # 结束同步音调
        t_end = np.linspace(0, sync_duration, sync_samples, endpoint=False)
        sync_end = 0.3 * np.sin(2 * np.pi * (self.base_freq + 2 * self.tone_spacing) * t_end)
        signal[-sync_samples:] = sync_end
        
        return signal

    def send_ft8_message(self, message, callsign1="CQ", callsign2="", grid=""):
        """
        发送FT8消息
        
        Args:
            message: 要发送的消息
            callsign1: 第一个呼号
            callsign2: 第二个呼号（可选）
            grid: 网格定位（可选）
        """
        print(f"发送FT8消息: {message}")
        
        # 编码消息
        symbols = self.encode_message(message, callsign1, callsign2, grid)
        print(f"编码符号: {symbols}")
        
        # 生成信号
        signal = self.generate_ft8_signal(symbols)
        
        # 添加同步音调
        signal = self.add_sync_tones(signal)
        
        # 播放信号
        sd.play(signal, self.sampling_rate)
        sd.wait()
        
        print("FT8消息发送完成")

    def detect_tones(self, signal, threshold=0.1):
        """
        检测信号中的音调
        
        Args:
            signal: 输入信号
            threshold: 检测阈值
            
        Returns:
            list: 检测到的音调序列
        """
        from scipy.fft import fft, fftfreq
        
        detected_tones = []
        samples_per_symbol = int(self.sampling_rate / self.symbol_rate)
        
        # 分块处理信号
        for i in range(0, len(signal) - samples_per_symbol, samples_per_symbol):
            chunk = signal[i:i + samples_per_symbol]
            
            # FFT分析
            fft_result = np.abs(fft(chunk))
            freqs = fftfreq(len(chunk), 1/self.sampling_rate)
            
            # 找到峰值频率
            peak_idx = np.argmax(fft_result[1:len(fft_result)//2]) + 1
            peak_freq = abs(freqs[peak_idx])
            
            # 匹配到最近的音调
            tone_idx = None
            min_diff = float('inf')
            
            for j, tone_freq in enumerate(self.tone_frequencies):
                diff = abs(peak_freq - tone_freq)
                if diff < min_diff and diff < self.tone_spacing / 2:
                    min_diff = diff
                    tone_idx = j
            
            if tone_idx is not None:
                detected_tones.append(tone_idx)
            else:
                detected_tones.append(0)  # 默认音调
        
        return detected_tones

    def decode_ft8_message(self, signal):
        """
        解码FT8消息
        
        Args:
            signal: 接收到的信号
            
        Returns:
            str: 解码后的消息
        """
        print("正在解码FT8消息...")
        
        # 检测音调
        tones = self.detect_tones(signal)
        print(f"检测到的音调: {tones}")
        
        # 简化的解码（实际FT8解码更复杂）
        message = ""
        for tone in tones:
            if tone < 26:
                message += chr(ord('A') + tone)
            elif tone < 36:
                message += str(tone - 26)
            elif tone == 36:
                message += " "
            elif tone == 37:
                message += "?"
            elif tone == 38:
                message += " "
            elif tone == 39:
                message += " "
        
        return message.strip()

    def receive_ft8_message(self, duration=15.0):
        """
        接收FT8消息
        
        Args:
            duration: 录音时长（秒）
            
        Returns:
            str: 接收到的消息
        """
        print("正在接收FT8消息...")
        
        # 录音
        signal = sd.rec(int(duration * self.sampling_rate), 
                       samplerate=self.sampling_rate, channels=1)
        sd.wait()
        
        # 解码
        message = self.decode_ft8_message(signal[:, 0])
        return message


if __name__ == "__main__":
    # 测试FT8调制器
    modulator = FT8Modulator()
    
    # 发送测试消息
    test_message = "HELLO WORLD"
    modulator.send_ft8_message(test_message, "CQ", "TEST", "FN42")
    
    # 接收测试（需要手动录音）
    # received_message = modulator.receive_ft8_message()
    # print(f"接收到的消息: {received_message}")
