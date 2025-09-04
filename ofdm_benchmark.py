#!/usr/bin/env python3
"""
OFDM性能基准测试和比较模块
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass

from ofdm_adaptive_qam import OFDMAdaptiveQAM, ModulationType
from advanced_ofdm import AdvancedOFDM, OFDMConfig

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    modulation_type: str
    spectral_efficiency: float
    bit_error_rate: float
    processing_time: float
    memory_usage: float
    snr_required: float
    complexity_score: int

class OFDMBenchmark:
    """OFDM性能基准测试类"""
    
    def __init__(self):
        self.results = {}
        self.test_data_sizes = [100, 500, 1000, 2000, 5000]  # 比特数
        self.snr_levels = [0, 5, 10, 15, 20, 25, 30]  # dB
        self.modulation_orders = [4, 16, 64, 256]
        
    def generate_test_data(self, size: int) -> np.ndarray:
        """生成测试数据"""
        return np.random.randint(0, 2, size)
    
    def add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """添加高斯白噪声"""
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 
                                           1j * np.random.randn(*signal.shape))
        return signal + noise
    
    def calculate_ber(self, original: np.ndarray, received: np.ndarray) -> float:
        """计算误码率"""
        min_len = min(len(original), len(received))
        if min_len == 0:
            return 1.0
        return np.mean(original[:min_len] != received[:min_len])
    
    def benchmark_adaptive_qam(self) -> Dict:
        """基准测试自适应QAM OFDM"""
        print("测试自适应QAM OFDM...")
        
        results = {
            'modulation_type': 'Adaptive QAM OFDM',
            'test_results': []
        }
        
        for data_size in self.test_data_sizes:
            for snr in self.snr_levels:
                # 生成测试数据
                test_bits = self.generate_test_data(data_size)
                
                # 创建调制器
                ofdm_mod = OFDMAdaptiveQAM(
                    sampling_rate=44100,
                    num_subcarriers=32,
                    num_pilot_carriers=4
                )
                
                # 设置SNR
                ofdm_mod.snr_estimate = snr
                
                # 测试调制
                start_time = time.time()
                signal = ofdm_mod.modulate_with_carrier(test_bits)
                modulate_time = time.time() - start_time
                
                # 添加噪声
                noisy_signal = self.add_noise(signal, snr)
                
                # 测试解调
                start_time = time.time()
                demodulated_bits, used_mod = ofdm_mod.demodulate_with_carrier(noisy_signal)
                demodulate_time = time.time() - start_time
                
                # 计算性能指标
                ber = self.calculate_ber(test_bits, demodulated_bits)
                spectral_efficiency = np.log2(used_mod.value)
                
                results['test_results'].append({
                    'data_size': data_size,
                    'snr_db': snr,
                    'ber': ber,
                    'spectral_efficiency': spectral_efficiency,
                    'modulate_time': modulate_time,
                    'demodulate_time': demodulate_time,
                    'total_time': modulate_time + demodulate_time,
                    'modulation_used': used_mod.name
                })
        
        return results
    
    def benchmark_advanced_ofdm(self) -> Dict:
        """基准测试高级OFDM"""
        print("测试高级OFDM...")
        
        results = {
            'modulation_type': 'Advanced OFDM',
            'test_results': []
        }
        
        # 创建配置
        config = OFDMConfig()
        config.num_subcarriers = 32
        config.num_pilot_carriers = 4
        config.use_fec = True
        config.use_sync = True
        
        for data_size in self.test_data_sizes:
            for snr in self.snr_levels:
                for mod_order in self.modulation_orders:
                    # 生成测试数据
                    test_bits = self.generate_test_data(data_size)
                    
                    # 创建调制器
                    ofdm_mod = AdvancedOFDM(config)
                    ofdm_mod.snr_estimate = snr
                    
                    # 测试调制
                    start_time = time.time()
                    signal = ofdm_mod.modulate(test_bits, modulation_order=mod_order, 
                                             use_advanced_features=True)
                    modulate_time = time.time() - start_time
                    
                    # 添加噪声
                    noisy_signal = self.add_noise(signal, snr)
                    
                    # 测试解调
                    start_time = time.time()
                    demodulated_bits, used_mod = ofdm_mod.demodulate(noisy_signal, 
                                                                    expected_modulation_order=mod_order,
                                                                    use_advanced_features=True)
                    demodulate_time = time.time() - start_time
                    
                    # 计算性能指标
                    ber = self.calculate_ber(test_bits, demodulated_bits)
                    spectral_efficiency = np.log2(mod_order)
                    
                    results['test_results'].append({
                        'data_size': data_size,
                        'snr_db': snr,
                        'modulation_order': mod_order,
                        'ber': ber,
                        'spectral_efficiency': spectral_efficiency,
                        'modulate_time': modulate_time,
                        'demodulate_time': demodulate_time,
                        'total_time': modulate_time + demodulate_time
                    })
        
        return results
    
    def compare_with_traditional_qam(self) -> Dict:
        """与传统QAM比较"""
        print("与传统QAM比较...")
        
        from qam_modulation import QAMModulator
        
        results = {
            'modulation_type': 'Traditional QAM',
            'test_results': []
        }
        
        for data_size in self.test_data_sizes:
            for snr in self.snr_levels:
                for mod_order in [16, 64, 256]:  # 传统QAM支持的阶数
                    # 生成测试数据
                    test_bits = self.generate_test_data(data_size)
                    
                    # 创建传统QAM调制器
                    qam_mod = QAMModulator(mod_order, 2000, 1000, 44100)
                    
                    # 测试调制
                    start_time = time.time()
                    signal = qam_mod.modulate_qam(test_bits)
                    modulate_time = time.time() - start_time
                    
                    # 添加噪声
                    noisy_signal = self.add_noise(signal, snr)
                    
                    # 测试解调
                    start_time = time.time()
                    demodulated_bits = qam_mod.demodulate_qam(noisy_signal)
                    demodulate_time = time.time() - start_time
                    
                    # 计算性能指标
                    ber = self.calculate_ber(test_bits, demodulated_bits)
                    spectral_efficiency = np.log2(mod_order)
                    
                    results['test_results'].append({
                        'data_size': data_size,
                        'snr_db': snr,
                        'modulation_order': mod_order,
                        'ber': ber,
                        'spectral_efficiency': spectral_efficiency,
                        'modulate_time': modulate_time,
                        'demodulate_time': demodulate_time,
                        'total_time': modulate_time + demodulate_time
                    })
        
        return results
    
    def run_all_benchmarks(self) -> Dict:
        """运行所有基准测试"""
        print("开始OFDM性能基准测试...")
        print("=" * 50)
        
        all_results = {}
        
        # 自适应QAM OFDM
        all_results['adaptive_qam'] = self.benchmark_adaptive_qam()
        
        # 高级OFDM
        all_results['advanced_ofdm'] = self.benchmark_advanced_ofdm()
        
        # 传统QAM
        all_results['traditional_qam'] = self.compare_with_traditional_qam()
        
        self.results = all_results
        return all_results
    
    def analyze_results(self) -> Dict:
        """分析测试结果"""
        if not self.results:
            print("请先运行基准测试")
            return {}
        
        analysis = {}
        
        for mod_type, results in self.results.items():
            test_results = results['test_results']
            
            # 计算平均性能指标
            avg_ber = np.mean([r['ber'] for r in test_results])
            avg_spectral_efficiency = np.mean([r['spectral_efficiency'] for r in test_results])
            avg_processing_time = np.mean([r['total_time'] for r in test_results])
            
            # 找到最佳SNR（BER < 1e-3）
            good_snr_results = [r for r in test_results if r['ber'] < 1e-3]
            min_snr = min([r['snr_db'] for r in good_snr_results]) if good_snr_results else float('inf')
            
            analysis[mod_type] = {
                'average_ber': avg_ber,
                'average_spectral_efficiency': avg_spectral_efficiency,
                'average_processing_time': avg_processing_time,
                'minimum_snr_for_good_performance': min_snr,
                'total_tests': len(test_results)
            }
        
        return analysis
    
    def generate_report(self, filename: str = 'ofdm_benchmark_report.json'):
        """生成测试报告"""
        if not self.results:
            print("请先运行基准测试")
            return
        
        # 分析结果
        analysis = self.analyze_results()
        
        # 生成报告
        report = {
            'test_configuration': {
                'data_sizes': self.test_data_sizes,
                'snr_levels': self.snr_levels,
                'modulation_orders': self.modulation_orders
            },
            'results': self.results,
            'analysis': analysis,
            'recommendations': self._generate_recommendations(analysis)
        }
        
        # 保存报告
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"测试报告已保存到: {filename}")
        
        # 打印摘要
        self._print_summary(analysis)
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 找到最佳性能的调制方式
        best_ber = min(analysis[mod]['average_ber'] for mod in analysis)
        best_mod = [mod for mod in analysis if analysis[mod]['average_ber'] == best_ber][0]
        
        recommendations.append(f"最佳误码率性能: {best_mod} (BER: {best_ber:.2e})")
        
        # 找到最高频谱效率
        best_eff = max(analysis[mod]['average_spectral_efficiency'] for mod in analysis)
        best_eff_mod = [mod for mod in analysis if analysis[mod]['average_spectral_efficiency'] == best_eff][0]
        
        recommendations.append(f"最高频谱效率: {best_eff_mod} ({best_eff:.2f} bits/symbol)")
        
        # 找到最快处理速度
        fastest = min(analysis[mod]['average_processing_time'] for mod in analysis)
        fastest_mod = [mod for mod in analysis if analysis[mod]['average_processing_time'] == fastest][0]
        
        recommendations.append(f"最快处理速度: {fastest_mod} ({fastest:.4f}s)")
        
        # 通用建议
        recommendations.append("建议:")
        recommendations.append("- 高SNR环境: 使用高级OFDM获得最佳性能")
        recommendations.append("- 低SNR环境: 使用自适应QAM OFDM提高可靠性")
        recommendations.append("- 实时应用: 考虑处理时间与性能的平衡")
        recommendations.append("- 频谱受限: 优先考虑频谱效率")
        
        return recommendations
    
    def _print_summary(self, analysis: Dict):
        """打印测试摘要"""
        print("\n" + "=" * 50)
        print("OFDM性能基准测试摘要")
        print("=" * 50)
        
        for mod_type, metrics in analysis.items():
            print(f"\n{mod_type.upper()}:")
            print(f"  平均误码率: {metrics['average_ber']:.2e}")
            print(f"  平均频谱效率: {metrics['average_spectral_efficiency']:.2f} bits/symbol")
            print(f"  平均处理时间: {metrics['average_processing_time']:.4f}s")
            print(f"  良好性能所需最小SNR: {metrics['minimum_snr_for_good_performance']:.1f} dB")
        
        print("\n建议:")
        for rec in self._generate_recommendations(analysis):
            print(f"  - {rec}")


def main():
    """主函数"""
    print("OFDM性能基准测试工具")
    print("=" * 50)
    
    # 创建基准测试器
    benchmark = OFDMBenchmark()
    
    # 运行所有测试
    results = benchmark.run_all_benchmarks()
    
    # 生成报告
    benchmark.generate_report()
    
    print("\n基准测试完成!")


if __name__ == "__main__":
    main()
