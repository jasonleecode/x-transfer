#!/usr/bin/env python3
"""
OFDM模块使用示例和接口说明
"""

import numpy as np
import time
from ofdm_adaptive_qam import OFDMAdaptiveQAM, ModulationType, OFDMAudioTransceiver
from advanced_ofdm import AdvancedOFDM, OFDMConfig

def example_adaptive_qam():
    """自适应QAM OFDM示例"""
    print("=== 自适应QAM OFDM示例 ===")
    
    # 创建OFDM调制器
    ofdm_mod = OFDMAdaptiveQAM(
        sampling_rate=44100,
        num_subcarriers=32,
        num_pilot_carriers=4,
        base_freq=2000,
        freq_spacing=200
    )
    
    # 创建音频收发器
    transceiver = OFDMAudioTransceiver(ofdm_mod)
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 1000)
    print(f"原始数据: {len(test_bits)} 比特")
    
    # 显示信道信息
    print("\n信道信息:")
    info = ofdm_mod.get_channel_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 调制
    print("\n调制中...")
    signal = ofdm_mod.modulate_with_carrier(test_bits)
    print(f"调制信号长度: {len(signal)} 样本")
    print(f"信号持续时间: {len(signal) / ofdm_mod.sampling_rate:.2f} 秒")
    
    # 解调
    print("\n解调中...")
    demodulated_bits, used_mod = ofdm_mod.demodulate_with_carrier(signal)
    print(f"解调数据: {len(demodulated_bits)} 比特")
    print(f"使用调制: {used_mod.name}")
    
    # 计算误码率
    if len(demodulated_bits) >= len(test_bits):
        error_rate = np.mean(demodulated_bits[:len(test_bits)] != test_bits)
        print(f"误码率: {error_rate:.4f}")
    
    return ofdm_mod, transceiver

def example_advanced_ofdm():
    """高级OFDM示例"""
    print("\n=== 高级OFDM示例 ===")
    
    # 创建配置
    config = OFDMConfig()
    config.num_subcarriers = 32
    config.num_pilot_carriers = 4
    config.use_fec = True
    config.use_sync = True
    config.use_precoding = True
    
    # 创建高级OFDM调制器
    ofdm_mod = AdvancedOFDM(config)
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 1000)
    print(f"原始数据: {len(test_bits)} 比特")
    
    # 调制
    print("\n高级OFDM调制中...")
    signal = ofdm_mod.modulate(test_bits, modulation_order=16, use_advanced_features=True)
    print(f"调制信号长度: {len(signal)} 样本")
    
    # 解调
    print("\n高级OFDM解调中...")
    demodulated_bits, used_mod = ofdm_mod.demodulate(signal, expected_modulation_order=16, 
                                                    use_advanced_features=True)
    print(f"解调数据: {len(demodulated_bits)} 比特")
    
    # 性能指标
    metrics = ofdm_mod.get_performance_metrics()
    print("\n性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return ofdm_mod

def example_audio_transmission():
    """音频传输示例"""
    print("\n=== 音频传输示例 ===")
    
    # 创建调制器和收发器
    ofdm_mod = OFDMAdaptiveQAM(sampling_rate=44100, num_subcarriers=16)
    transceiver = OFDMAudioTransceiver(ofdm_mod)
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 500)
    print(f"发送数据: {len(test_bits)} 比特")
    
    # 发送
    print("发送中...")
    success = transceiver.transmit_bits(test_bits, ModulationType.QAM16)
    
    if success:
        print("发送成功!")
        
        # 接收
        print("接收中...")
        received_bits, used_mod = transceiver.receive_bits(duration=5.0)
        
        if received_bits is not None:
            print(f"接收数据: {len(received_bits)} 比特")
            print(f"使用调制: {used_mod.name}")
            
            # 计算误码率
            if len(received_bits) >= len(test_bits):
                error_rate = np.mean(received_bits[:len(test_bits)] != test_bits)
                print(f"误码率: {error_rate:.4f}")
        else:
            print("接收失败")
    else:
        print("发送失败")

def example_performance_comparison():
    """性能比较示例"""
    print("\n=== 性能比较示例 ===")
    
    # 测试数据
    test_bits = np.random.randint(0, 2, 1000)
    
    # 自适应QAM OFDM
    print("测试自适应QAM OFDM...")
    ofdm_adaptive = OFDMAdaptiveQAM(sampling_rate=44100, num_subcarriers=32)
    
    start_time = time.time()
    signal_adaptive = ofdm_adaptive.modulate_with_carrier(test_bits)
    modulate_time_adaptive = time.time() - start_time
    
    start_time = time.time()
    demod_adaptive, _ = ofdm_adaptive.demodulate_with_carrier(signal_adaptive)
    demodulate_time_adaptive = time.time() - start_time
    
    # 高级OFDM
    print("测试高级OFDM...")
    config = OFDMConfig()
    config.num_subcarriers = 32
    ofdm_advanced = AdvancedOFDM(config)
    
    start_time = time.time()
    signal_advanced = ofdm_advanced.modulate(test_bits, modulation_order=16, use_advanced_features=True)
    modulate_time_advanced = time.time() - start_time
    
    start_time = time.time()
    demod_advanced, _ = ofdm_advanced.demodulate(signal_advanced, expected_modulation_order=16, 
                                                use_advanced_features=True)
    demodulate_time_advanced = time.time() - start_time
    
    # 比较结果
    print("\n性能比较结果:")
    print(f"{'指标':<20} {'自适应QAM':<15} {'高级OFDM':<15}")
    print("-" * 50)
    print(f"{'调制时间(s)':<20} {modulate_time_adaptive:<15.4f} {modulate_time_advanced:<15.4f}")
    print(f"{'解调时间(s)':<20} {demodulate_time_adaptive:<15.4f} {demodulate_time_advanced:<15.4f}")
    print(f"{'总时间(s)':<20} {modulate_time_adaptive + demodulate_time_adaptive:<15.4f} {modulate_time_advanced + demodulate_time_advanced:<15.4f}")
    print(f"{'信号长度':<20} {len(signal_adaptive):<15} {len(signal_advanced):<15}")
    
    # 误码率比较
    if len(demod_adaptive) >= len(test_bits):
        ber_adaptive = np.mean(demod_adaptive[:len(test_bits)] != test_bits)
        print(f"{'误码率':<20} {ber_adaptive:<15.4f}", end="")
    else:
        print(f"{'误码率':<20} {'N/A':<15}", end="")
    
    if len(demod_advanced) >= len(test_bits):
        ber_advanced = np.mean(demod_advanced[:len(test_bits)] != test_bits)
        print(f" {ber_advanced:<15.4f}")
    else:
        print(f" {'N/A':<15}")

def example_modulation_adaptation():
    """调制自适应示例"""
    print("\n=== 调制自适应示例 ===")
    
    ofdm_mod = OFDMAdaptiveQAM(sampling_rate=44100, num_subcarriers=32)
    
    # 测试不同SNR下的调制选择
    snr_levels = [5, 10, 15, 20, 25, 30]
    test_bits = np.random.randint(0, 2, 500)
    
    print("SNR (dB) | 选择的调制方式 | 频谱效率 (bits/symbol)")
    print("-" * 50)
    
    for snr in snr_levels:
        ofdm_mod.snr_estimate = snr
        modulation = ofdm_mod._select_modulation(snr)
        efficiency = ofdm_mod.get_spectral_efficiency(modulation)
        
        print(f"{snr:8} | {modulation.name:12} | {efficiency:20}")
    
    # 测试自适应调制
    print("\n测试自适应调制...")
    ofdm_mod.set_adaptive_modulation(True)
    
    signal = ofdm_mod.modulate_with_carrier(test_bits)
    demodulated_bits, used_mod = ofdm_mod.demodulate_with_carrier(signal)
    
    print(f"自适应选择的调制: {used_mod.name}")
    print(f"频谱效率: {ofdm_mod.get_spectral_efficiency(used_mod)} bits/symbol")

def example_power_control():
    """功率控制示例"""
    print("\n=== 功率控制示例 ===")
    
    ofdm_mod = OFDMAdaptiveQAM(sampling_rate=44100, num_subcarriers=32)
    
    # 测试不同调制方式的功率分配
    modulations = [ModulationType.QPSK, ModulationType.QAM16, 
                   ModulationType.QAM64, ModulationType.QAM256]
    
    print("调制方式 | 功率分配")
    print("-" * 30)
    
    for mod in modulations:
        power_allocation = ofdm_mod._calculate_power_allocation(mod)
        avg_power = np.mean(power_allocation)
        print(f"{mod.name:8} | {avg_power:.3f}")

def main():
    """主函数 - 运行所有示例"""
    print("OFDM模块使用示例")
    print("=" * 50)
    
    try:
        # 基本示例
        example_adaptive_qam()
        example_advanced_ofdm()
        
        # 性能比较
        example_performance_comparison()
        
        # 高级功能
        example_modulation_adaptation()
        example_power_control()
        
        # 音频传输（需要用户交互）
        print("\n是否运行音频传输示例？(需要音频设备)")
        response = input("输入 'y' 继续，其他键跳过: ").lower()
        if response == 'y':
            example_audio_transmission()
        
        print("\n所有示例运行完成!")
        
    except Exception as e:
        print(f"运行示例时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
