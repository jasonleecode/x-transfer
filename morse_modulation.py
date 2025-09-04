import numpy as np
import sounddevice as sd

class MorseModulator:
    """
    摩尔斯电码调制器
    """
    def __init__(self, dot_duration=0.1, sampling_rate=44100):
        """
        初始化摩尔斯调制器
        
        Args:
            dot_duration: 点的时间长度（秒）
            sampling_rate: 采样率（Hz）
        """
        self.dot_duration = dot_duration
        self.dash_duration = dot_duration * 3
        self.symbol_gap = dot_duration
        self.letter_gap = dot_duration * 3
        self.word_gap = dot_duration * 7
        self.sampling_rate = sampling_rate
        
        # 摩尔斯电码表
        self.morse_code = {
            'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
            'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
            'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
            'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
            'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
            'Z': '--..',
            '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
            '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
            ' ': ' ', '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.',
            '!': '-.-.--', '/': '-..-.', '(': '-.--.', ')': '-.--.-', '&': '.-...',
            ':': '---...', ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-',
            '_': '..--.-', '"': '.-..-.', '$': '...-..-', '@': '.--.-.'
        }
        
        # 反向查找表
        self.reverse_morse = {v: k for k, v in self.morse_code.items()}

    def text_to_morse(self, text):
        """
        将文本转换为摩尔斯电码
        
        Args:
            text: 输入文本
            
        Returns:
            str: 摩尔斯电码字符串
        """
        morse = []
        for char in text.upper():
            if char in self.morse_code:
                morse.append(self.morse_code[char])
            else:
                morse.append(' ')  # 未知字符用空格代替
        return ' '.join(morse)

    def morse_to_signal(self, morse_text, frequency=1000):
        """
        将摩尔斯电码转换为音频信号
        
        Args:
            morse_text: 摩尔斯电码字符串
            frequency: 载波频率（Hz）
            
        Returns:
            numpy.ndarray: 音频信号
        """
        signal = []
        
        for char in morse_text:
            if char == '.':
                # 点：短音
                dot_signal = self.generate_tone(frequency, self.dot_duration)
                signal.extend(dot_signal)
                # 符号间隔
                gap = np.zeros(int(self.symbol_gap * self.sampling_rate))
                signal.extend(gap)
                
            elif char == '-':
                # 划：长音
                dash_signal = self.generate_tone(frequency, self.dash_duration)
                signal.extend(dash_signal)
                # 符号间隔
                gap = np.zeros(int(self.symbol_gap * self.sampling_rate))
                signal.extend(gap)
                
            elif char == ' ':
                # 字母间隔
                gap = np.zeros(int(self.letter_gap * self.sampling_rate))
                signal.extend(gap)
                
            elif char == '  ':
                # 单词间隔
                gap = np.zeros(int(self.word_gap * self.sampling_rate))
                signal.extend(gap)
        
        return np.array(signal)

    def generate_tone(self, frequency, duration):
        """
        生成指定频率和持续时间的音频信号
        
        Args:
            frequency: 频率（Hz）
            duration: 持续时间（秒）
            
        Returns:
            numpy.ndarray: 音频信号
        """
        t = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        return 0.5 * np.sin(2 * np.pi * frequency * t)

    def signal_to_morse(self, signal, frequency=1000, threshold=0.1):
        """
        将音频信号解码为摩尔斯电码
        
        Args:
            signal: 音频信号
            frequency: 载波频率（Hz）
            threshold: 检测阈值
            
        Returns:
            str: 摩尔斯电码字符串
        """
        # 简单的能量检测
        window_size = int(self.sampling_rate * self.dot_duration / 4)
        energy = []
        
        for i in range(0, len(signal) - window_size, window_size):
            window = signal[i:i + window_size]
            energy.append(np.mean(window ** 2))
        
        # 检测音调和静音
        morse = []
        in_tone = False
        tone_length = 0
        silence_length = 0
        
        for e in energy:
            if e > threshold:
                if not in_tone:
                    in_tone = True
                    if silence_length > 0:
                        # 根据静音长度判断间隔类型
                        if silence_length > 5:  # 单词间隔
                            morse.append('  ')
                        elif silence_length > 2:  # 字母间隔
                            morse.append(' ')
                tone_length += 1
                silence_length = 0
            else:
                if in_tone:
                    in_tone = False
                    # 根据音调长度判断点或划
                    if tone_length > 2:
                        morse.append('-')
                    else:
                        morse.append('.')
                tone_length = 0
                silence_length += 1
        
        return ''.join(morse)

    def morse_to_text(self, morse_text):
        """
        将摩尔斯电码转换为文本
        
        Args:
            morse_text: 摩尔斯电码字符串
            
        Returns:
            str: 解码后的文本
        """
        words = morse_text.split('  ')  # 按单词分割
        decoded_words = []
        
        for word in words:
            letters = word.split()  # 按字母分割
            decoded_letters = []
            for letter in letters:
                if letter in self.reverse_morse:
                    decoded_letters.append(self.reverse_morse[letter])
            decoded_words.append(''.join(decoded_letters))
        
        return ' '.join(decoded_words)

    def send_text(self, text, frequency=1000):
        """
        发送文本（转换为摩尔斯电码并播放）
        
        Args:
            text: 要发送的文本
            frequency: 载波频率（Hz）
        """
        morse = self.text_to_morse(text)
        print(f"摩尔斯电码: {morse}")
        
        signal = self.morse_to_signal(morse, frequency)
        sd.play(signal, self.sampling_rate)
        sd.wait()

    def receive_text(self, duration, frequency=1000):
        """
        接收摩尔斯电码并解码为文本
        
        Args:
            duration: 录音时长（秒）
            frequency: 载波频率（Hz）
            
        Returns:
            str: 解码后的文本
        """
        print("正在接收摩尔斯电码...")
        signal = sd.rec(int(duration * self.sampling_rate), 
                       samplerate=self.sampling_rate, channels=1)
        sd.wait()
        
        morse = self.signal_to_morse(signal[:, 0], frequency)
        print(f"接收到的摩尔斯电码: {morse}")
        
        text = self.morse_to_text(morse)
        return text


if __name__ == "__main__":
    # 测试摩尔斯调制器
    modulator = MorseModulator()
    
    # 发送测试
    test_text = "HELLO WORLD"
    print(f"发送文本: {test_text}")
    modulator.send_text(test_text)
    
    # 接收测试（需要手动录音）
    # received_text = modulator.receive_text(10)
    # print(f"接收到的文本: {received_text}")
