#!/usr/bin/env python3
"""
实时QAM接收测试脚本
"""

import sys
import time
from realtime_qam_receiver import RealtimeQAMReceiver

def test_progress_callback(progress, received_bits=0, expected_bits=0):
    """测试进度回调函数"""
    if expected_bits > 0:
        print(f"\r接收进度: {progress:.1f}% ({received_bits}/{expected_bits} 比特)", end='', flush=True)
    else:
        print(f"\r接收进度: {progress:.1f}%", end='', flush=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_realtime.py <output_file>")
        print("Example: python test_realtime.py received_file.txt")
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    print("=== 实时QAM接收测试 ===")
    print(f"输出文件: {output_file}")
    print("按 Ctrl+C 停止接收")
    print()
    
    # 创建接收器
    receiver = RealtimeQAMReceiver()
    receiver.set_progress_callback(test_progress_callback)
    
    try:
        # 开始接收
        print("开始接收...")
        success = receiver.start_receiving(output_file)
        
        if success:
            print("\n✅ 文件接收成功!")
        else:
            print("\n❌ 文件接收失败!")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断接收")
        receiver.stop_receiving()
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        receiver.stop_receiving()

if __name__ == "__main__":
    main()
