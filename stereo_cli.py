#!/usr/bin/env python3
"""
双声道传输命令行工具
"""

import sys
import argparse
import time
from stereo_transmission import StereoTransmission, StereoMode, StereoFileTransmission

def send_file_mode(args):
    """发送文件模式"""
    print(f"发送文件: {args.file}")
    print(f"模式: {args.mode}")
    
    # 创建传输器
    stereo_transmission = StereoTransmission(
        sampling_rate=args.sample_rate,
        carrier_freq=args.carrier_freq,
        symbol_rate=args.symbol_rate,
        modulation_order=args.modulation,
        mode=StereoMode(args.mode)
    )
    
    # 创建文件传输器
    file_transmission = StereoFileTransmission(stereo_transmission)
    
    # 发送文件
    success = file_transmission.send_file(args.file, chunk_size=args.chunk_size)
    
    if success:
        print("文件发送成功!")
        return 0
    else:
        print("文件发送失败!")
        return 1

def receive_file_mode(args):
    """接收文件模式"""
    print(f"接收文件: {args.output}")
    print(f"模式: {args.mode}")
    print(f"超时时间: {args.timeout} 秒")
    
    # 创建传输器
    stereo_transmission = StereoTransmission(
        sampling_rate=args.sample_rate,
        carrier_freq=args.carrier_freq,
        symbol_rate=args.symbol_rate,
        modulation_order=args.modulation,
        mode=StereoMode(args.mode)
    )
    
    # 创建文件传输器
    file_transmission = StereoFileTransmission(stereo_transmission)
    
    # 接收文件
    success = file_transmission.receive_file(args.output, duration=args.timeout)
    
    if success:
        print("文件接收成功!")
        return 0
    else:
        print("文件接收失败!")
        return 1

def test_mode(args):
    """测试模式"""
    print("双声道传输测试")
    print("=" * 30)
    
    # 创建传输器
    stereo_transmission = StereoTransmission(
        sampling_rate=args.sample_rate,
        carrier_freq=args.carrier_freq,
        symbol_rate=args.symbol_rate,
        modulation_order=args.modulation,
        mode=StereoMode(args.mode)
    )
    
    # 显示配置信息
    info = stereo_transmission.get_transmission_info()
    print("配置信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试数据
    import numpy as np
    test_bits = np.random.randint(0, 2, 1000)
    print(f"\n测试数据: {len(test_bits)} 比特")
    
    # 调制测试
    print("\n调制测试...")
    start_time = time.time()
    stereo_signal = stereo_transmission.modulate_stereo(test_bits)
    modulate_time = time.time() - start_time
    print(f"调制时间: {modulate_time:.4f} 秒")
    print(f"立体声信号形状: {stereo_signal.shape}")
    
    # 解调测试
    print("\n解调测试...")
    start_time = time.time()
    demodulated_bits = stereo_transmission.demodulate_stereo(stereo_signal)
    demodulate_time = time.time() - start_time
    print(f"解调时间: {demodulate_time:.4f} 秒")
    print(f"解调数据长度: {len(demodulated_bits)}")
    
    # 误码率测试
    if len(demodulated_bits) >= len(test_bits):
        error_rate = np.mean(demodulated_bits[:len(test_bits)] != test_bits)
        print(f"误码率: {error_rate:.4f}")
    else:
        print("数据长度不匹配")
    
    # 音频传输测试
    if args.audio_test:
        print("\n音频传输测试...")
        print("发送测试数据...")
        success = stereo_transmission.transmit_data(test_bits, use_sync=True)
        
        if success:
            print("发送成功!")
            print("接收测试数据...")
            received_bits = stereo_transmission.receive_data(duration=10, use_sync=True)
            
            if received_bits is not None:
                print(f"接收数据: {len(received_bits)} 比特")
                if len(received_bits) >= len(test_bits):
                    error_rate = np.mean(received_bits[:len(test_bits)] != test_bits)
                    print(f"音频传输误码率: {error_rate:.4f}")
            else:
                print("接收失败")
        else:
            print("发送失败")
    
    return 0

def benchmark_mode(args):
    """基准测试模式"""
    print("双声道传输基准测试")
    print("=" * 30)
    
    import numpy as np
    from qam_modulation import QAMModulator
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 1000)
    
    # 单声道基准
    print("单声道基准测试...")
    single_modulator = QAMModulator(args.modulation, args.carrier_freq, args.symbol_rate, args.sample_rate)
    
    start_time = time.time()
    single_signal = single_modulator.modulate_qam(test_bits)
    single_modulate_time = time.time() - start_time
    
    start_time = time.time()
    single_demodulated = single_modulator.demodulate_qam(single_signal)
    single_demodulate_time = time.time() - start_time
    
    # 双声道测试
    print("双声道测试...")
    stereo_transmission = StereoTransmission(
        sampling_rate=args.sample_rate,
        carrier_freq=args.carrier_freq,
        symbol_rate=args.symbol_rate,
        modulation_order=args.modulation,
        mode=StereoMode(args.mode)
    )
    
    start_time = time.time()
    stereo_signal = stereo_transmission.modulate_stereo(test_bits)
    stereo_modulate_time = time.time() - start_time
    
    start_time = time.time()
    stereo_demodulated = stereo_transmission.demodulate_stereo(stereo_signal)
    stereo_demodulate_time = time.time() - start_time
    
    # 比较结果
    print("\n性能比较:")
    print(f"{'指标':<20} {'单声道':<15} {'双声道':<15} {'提升':<15}")
    print("-" * 65)
    print(f"{'调制时间(s)':<20} {single_modulate_time:<15.4f} {stereo_modulate_time:<15.4f} {single_modulate_time/stereo_modulate_time:<15.2f}x")
    print(f"{'解调时间(s)':<20} {single_demodulate_time:<15.4f} {stereo_demodulate_time:<15.4f} {single_demodulate_time/stereo_demodulate_time:<15.2f}x")
    print(f"{'总时间(s)':<20} {single_modulate_time + single_demodulate_time:<15.4f} {stereo_modulate_time + stereo_demodulate_time:<15.4f} {(single_modulate_time + single_demodulate_time)/(stereo_modulate_time + stereo_demodulate_time):<15.2f}x")
    print(f"{'信号长度':<20} {len(single_signal):<15} {stereo_signal.shape[0]:<15} {len(single_signal)/stereo_signal.shape[0]:<15.2f}x")
    
    # 误码率比较
    single_ber = np.mean(single_demodulated[:len(test_bits)] != test_bits)
    stereo_ber = np.mean(stereo_demodulated[:len(test_bits)] != test_bits)
    print(f"{'误码率':<20} {single_ber:<15.4f} {stereo_ber:<15.4f} {single_ber/stereo_ber if stereo_ber > 0 else 'N/A':<15}")
    
    return 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='双声道音频传输工具')
    
    # 通用参数
    parser.add_argument('--sample-rate', type=int, default=44100, help='采样率 (Hz)')
    parser.add_argument('--carrier-freq', type=int, default=2000, help='载波频率 (Hz)')
    parser.add_argument('--symbol-rate', type=int, default=1000, help='符号率 (Hz)')
    parser.add_argument('--modulation', type=int, default=16, choices=[4, 16, 64, 256], help='调制阶数')
    parser.add_argument('--mode', type=str, default='parallel', 
                       choices=['parallel', 'backup', 'mixed'], help='传输模式')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 发送文件命令
    send_parser = subparsers.add_parser('send', help='发送文件')
    send_parser.add_argument('file', help='要发送的文件')
    send_parser.add_argument('--chunk-size', type=int, default=1024, help='块大小')
    
    # 接收文件命令
    receive_parser = subparsers.add_parser('receive', help='接收文件')
    receive_parser.add_argument('output', help='输出文件路径')
    receive_parser.add_argument('--timeout', type=int, default=30, help='超时时间 (秒)')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='运行测试')
    test_parser.add_argument('--audio-test', action='store_true', help='包含音频传输测试')
    
    # 基准测试命令
    benchmark_parser = subparsers.add_parser('benchmark', help='运行基准测试')
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'send':
            return send_file_mode(args)
        elif args.command == 'receive':
            return receive_file_mode(args)
        elif args.command == 'test':
            return test_mode(args)
        elif args.command == 'benchmark':
            return benchmark_mode(args)
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n用户中断")
        return 1
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
