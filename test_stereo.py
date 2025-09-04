#!/usr/bin/env python3
"""
双声道传输测试脚本
"""

import numpy as np
import time
import sys
import os
from stereo_transmission import StereoTransmission, StereoMode, StereoFileTransmission

def test_basic_transmission():
    """测试基础传输功能"""
    print("=== 基础双声道传输测试 ===")
    
    # 创建传输器
    stereo_transmission = StereoTransmission(
        sampling_rate=44100,
        carrier_freq=2000,
        symbol_rate=1000,
        modulation_order=16,
        mode=StereoMode.PARALLEL
    )
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 500)
    print(f"测试数据: {len(test_bits)} 比特")
    
    # 测试不同模式
    modes = [StereoMode.PARALLEL, StereoMode.BACKUP, StereoMode.MIXED]
    
    for mode in modes:
        print(f"\n测试模式: {mode.value}")
        stereo_transmission.set_mode(mode)
        
        # 调制
        stereo_signal = stereo_transmission.modulate_stereo(test_bits)
        print(f"立体声信号形状: {stereo_signal.shape}")
        
        # 解调
        demodulated_bits = stereo_transmission.demodulate_stereo(stereo_signal)
        print(f"解调数据长度: {len(demodulated_bits)}")
        
        # 计算误码率
        if len(demodulated_bits) >= len(test_bits):
            error_rate = np.mean(demodulated_bits[:len(test_bits)] != test_bits)
            print(f"误码率: {error_rate:.4f}")
        else:
            print("数据长度不匹配")

def test_audio_transmission():
    """测试音频传输"""
    print("\n=== 音频传输测试 ===")
    
    stereo_transmission = StereoTransmission(
        sampling_rate=44100,
        carrier_freq=2000,
        symbol_rate=1000,
        modulation_order=16,
        mode=StereoMode.PARALLEL
    )
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 200)
    print(f"发送数据: {len(test_bits)} 比特")
    
    # 发送
    print("发送中...")
    success = stereo_transmission.transmit_data(test_bits, use_sync=True)
    
    if success:
        print("发送成功!")
        
        # 接收
        print("接收中...")
        received_bits = stereo_transmission.receive_data(duration=10, use_sync=True)
        
        if received_bits is not None:
            print(f"接收数据: {len(received_bits)} 比特")
            
            # 计算误码率
            if len(received_bits) >= len(test_bits):
                error_rate = np.mean(received_bits[:len(test_bits)] != test_bits)
                print(f"误码率: {error_rate:.4f}")
            else:
                print("数据长度不匹配")
        else:
            print("接收失败")
    else:
        print("发送失败")

def test_file_transmission():
    """测试文件传输"""
    print("\n=== 文件传输测试 ===")
    
    # 创建测试文件
    test_file = "test_stereo.txt"
    test_content = "Hello, Stereo Transmission! This is a test file for dual-channel audio transmission."
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    print(f"创建测试文件: {test_file}")
    
    # 创建文件传输器
    stereo_transmission = StereoTransmission(
        sampling_rate=44100,
        carrier_freq=2000,
        symbol_rate=1000,
        modulation_order=16,
        mode=StereoMode.PARALLEL
    )
    
    file_transmission = StereoFileTransmission(stereo_transmission)
    
    # 发送文件
    print("发送文件...")
    success = file_transmission.send_file(test_file)
    
    if success:
        print("文件发送成功!")
        
        # 接收文件
        print("接收文件...")
        received_file = "received_stereo.txt"
        success = file_transmission.receive_file(received_file, duration=30)
        
        if success:
            print("文件接收成功!")
            
            # 验证文件内容
            with open(received_file, 'r') as f:
                received_content = f.read()
            
            if received_content == test_content:
                print("文件内容验证成功!")
            else:
                print("文件内容验证失败")
        else:
            print("文件接收失败")
    else:
        print("文件发送失败")
    
    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists("received_stereo.txt"):
        os.remove("received_stereo.txt")

def test_performance_comparison():
    """性能比较测试"""
    print("\n=== 性能比较测试 ===")
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 1000)
    
    # 单声道传输（模拟）
    print("单声道传输测试...")
    from qam_modulation import QAMModulator
    
    single_modulator = QAMModulator(16, 2000, 1000, 44100)
    
    start_time = time.time()
    single_signal = single_modulator.modulate_qam(test_bits)
    single_modulate_time = time.time() - start_time
    
    start_time = time.time()
    single_demodulated = single_modulator.demodulate_qam(single_signal)
    single_demodulate_time = time.time() - start_time
    
    # 双声道传输
    print("双声道传输测试...")
    stereo_transmission = StereoTransmission(
        sampling_rate=44100,
        carrier_freq=2000,
        symbol_rate=1000,
        modulation_order=16,
        mode=StereoMode.PARALLEL
    )
    
    start_time = time.time()
    stereo_signal = stereo_transmission.modulate_stereo(test_bits)
    stereo_modulate_time = time.time() - start_time
    
    start_time = time.time()
    stereo_demodulated = stereo_transmission.demodulate_stereo(stereo_signal)
    stereo_demodulate_time = time.time() - start_time
    
    # 比较结果
    print("\n性能比较结果:")
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

def test_mode_comparison():
    """模式比较测试"""
    print("\n=== 模式比较测试 ===")
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 1000)
    
    modes = [StereoMode.PARALLEL, StereoMode.BACKUP, StereoMode.MIXED]
    
    print(f"{'模式':<15} {'信号长度':<15} {'误码率':<15} {'处理时间(s)':<15}")
    print("-" * 60)
    
    for mode in modes:
        stereo_transmission = StereoTransmission(
            sampling_rate=44100,
            carrier_freq=2000,
            symbol_rate=1000,
            modulation_order=16,
            mode=mode
        )
        
        start_time = time.time()
        stereo_signal = stereo_transmission.modulate_stereo(test_bits)
        demodulated_bits = stereo_transmission.demodulate_stereo(stereo_signal)
        total_time = time.time() - start_time
        
        error_rate = np.mean(demodulated_bits[:len(test_bits)] != test_bits)
        
        print(f"{mode.value:<15} {stereo_signal.shape[0]:<15} {error_rate:<15.4f} {total_time:<15.4f}")

def main():
    """主函数"""
    print("双声道传输测试工具")
    print("=" * 50)
    
    try:
        # 基础功能测试
        test_basic_transmission()
        
        # 性能比较
        test_performance_comparison()
        
        # 模式比较
        test_mode_comparison()
        
        # 文件传输测试
        test_file_transmission()
        
        # 音频传输测试（需要用户确认）
        print("\n是否运行音频传输测试？(需要音频设备)")
        response = input("输入 'y' 继续，其他键跳过: ").lower()
        if response == 'y':
            test_audio_transmission()
        
        print("\n所有测试完成!")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
