#!/usr/bin/env python3
"""
双声道传输调试脚本
"""

import numpy as np
from stereo_transmission import StereoTransmission, StereoMode
from qam_modulation import QAMModulator

def debug_data_split_merge():
    """调试数据分割和合并"""
    print("=== 数据分割和合并调试 ===")
    
    # 测试数据
    test_data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    print(f"原始数据: {test_data}")
    
    # 创建传输器
    stereo_transmission = StereoTransmission(mode=StereoMode.PARALLEL)
    
    # 分割数据
    left_data, right_data = stereo_transmission.split_data(test_data)
    print(f"左声道数据: {left_data}")
    print(f"右声道数据: {right_data}")
    
    # 合并数据
    merged_data = stereo_transmission.merge_data(left_data, right_data)
    print(f"合并数据: {merged_data}")
    
    # 检查是否匹配
    if len(merged_data) >= len(test_data):
        match = np.array_equal(merged_data[:len(test_data)], test_data)
        print(f"数据匹配: {match}")
    else:
        print("数据长度不匹配")

def debug_qam_modulation():
    """调试QAM调制解调"""
    print("\n=== QAM调制解调调试 ===")
    
    # 测试数据
    test_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    print(f"原始比特: {test_bits}")
    
    # 创建QAM调制器
    qam_mod = QAMModulator(16, 2000, 1000, 44100)
    
    # 调制
    signal = qam_mod.modulate_qam(test_bits)
    print(f"调制信号长度: {len(signal)}")
    
    # 解调
    demodulated_bits = qam_mod.demodulate_qam(signal)
    print(f"解调比特: {demodulated_bits}")
    print(f"解调比特长度: {len(demodulated_bits)}")
    
    # 检查匹配
    if len(demodulated_bits) >= len(test_bits):
        match = np.array_equal(demodulated_bits[:len(test_bits)], test_bits)
        print(f"QAM匹配: {match}")
        if not match:
            print(f"错误比特: {demodulated_bits[:len(test_bits)] != test_bits}")
    else:
        print("QAM长度不匹配")

def debug_stereo_modulation():
    """调试双声道调制解调"""
    print("\n=== 双声道调制解调调试 ===")
    
    # 测试数据
    test_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    print(f"原始比特: {test_bits}")
    
    # 创建双声道传输器
    stereo_transmission = StereoTransmission(mode=StereoMode.PARALLEL)
    
    # 调制
    stereo_signal = stereo_transmission.modulate_stereo(test_bits)
    print(f"立体声信号形状: {stereo_signal.shape}")
    
    # 解调
    demodulated_bits = stereo_transmission.demodulate_stereo(stereo_signal)
    print(f"解调比特: {demodulated_bits}")
    print(f"解调比特长度: {len(demodulated_bits)}")
    
    # 检查匹配
    if len(demodulated_bits) >= len(test_bits):
        match = np.array_equal(demodulated_bits[:len(test_bits)], test_bits)
        print(f"双声道匹配: {match}")
        if not match:
            print(f"错误比特: {demodulated_bits[:len(test_bits)] != test_bits}")
    else:
        print("双声道长度不匹配")

def debug_step_by_step():
    """逐步调试"""
    print("\n=== 逐步调试 ===")
    
    # 测试数据
    test_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    print(f"原始比特: {test_bits}")
    
    # 创建传输器
    stereo_transmission = StereoTransmission(mode=StereoMode.PARALLEL)
    
    # 步骤1：分割数据
    left_data, right_data = stereo_transmission.split_data(test_bits)
    print(f"分割后 - 左: {left_data}, 右: {right_data}")
    
    # 步骤2：调制左右声道
    left_signal = stereo_transmission.left_modulator.modulate_qam(left_data)
    right_signal = stereo_transmission.right_modulator.modulate_qam(right_data)
    print(f"调制后 - 左信号长度: {len(left_signal)}, 右信号长度: {len(right_signal)}")
    
    # 步骤3：解调左右声道
    left_demod = stereo_transmission.left_modulator.demodulate_qam(left_signal)
    right_demod = stereo_transmission.right_modulator.demodulate_qam(right_signal)
    print(f"解调后 - 左: {left_demod}, 右: {right_demod}")
    
    # 步骤4：合并数据
    merged_data = stereo_transmission.merge_data(left_demod, right_demod)
    print(f"合并后: {merged_data}")
    
    # 检查匹配
    if len(merged_data) >= len(test_bits):
        match = np.array_equal(merged_data[:len(test_bits)], test_bits)
        print(f"最终匹配: {match}")
        if not match:
            print(f"错误比特: {merged_data[:len(test_bits)] != test_bits}")
    else:
        print("最终长度不匹配")

if __name__ == "__main__":
    debug_data_split_merge()
    debug_qam_modulation()
    debug_stereo_modulation()
    debug_step_by_step()
