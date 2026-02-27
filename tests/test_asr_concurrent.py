"""
RealtimeFunASR 并发测试

测试 RealtimeFunASR 是否支持 10 个并发连接
"""
import os
import sys
import time
import threading
import wave
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from asr.RealtimeFunASR import RealtimeFunASR
from asr.utils.asr_postprocessor import postprocess_asr
from utils.log import logger


class ConcurrentASRTester:
    """并发ASR测试器"""
    
    def __init__(self, audio_file: str, num_concurrent: int = 10):
        """
        初始化测试器
        
        Args:
            audio_file: 测试音频文件路径
            num_concurrent: 并发数量
        """
        self.audio_file = audio_file
        self.num_concurrent = num_concurrent
        self.results: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
    def load_audio_chunks(self) -> List[bytes]:
        """
        加载音频文件并切分成chunks
        
        Returns:
            音频数据块列表
        """
        chunks = []
        try:
            with wave.open(self.audio_file, 'rb') as wf:
                # 读取音频参数
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                
                logger.info(f"音频参数: channels={channels}, sample_width={sample_width}, framerate={framerate}")
                
                # 每次读取 320 帧 (20ms @ 16kHz)
                chunk_frames = 320
                
                while True:
                    data = wf.readframes(chunk_frames)
                    if not data:
                        break
                    chunks.append(data)
                    
            logger.info(f"加载音频完成，共 {len(chunks)} 个chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"加载音频文件失败: {e}")
            raise
    
    def single_asr_task(self, task_id: int, audio_chunks: List[bytes]) -> Dict[str, Any]:
        """
        单个ASR任务
        
        Args:
            task_id: 任务ID
            audio_chunks: 音频数据块
            
        Returns:
            任务结果字典
        """
        result = {
            "task_id": task_id,
            "success": False,
            "raw_text": "",
            "processed_text": "",
            "start_time": 0,
            "end_time": 0,
            "duration": 0,
            "error": None,
            "events": []
        }
        
        try:
            logger.info(f"[任务 {task_id}] 开始识别...")
            result["start_time"] = time.time()
            
            # 创建ASR实例
            asr = RealtimeFunASR()
            
            # 收集识别结果
            intermediate_texts = []
            final_text = ""
            
            def on_result(event: str, text: str):
                """结果回调"""
                result["events"].append({"event": event, "text": text, "time": time.time()})
                
                if event == "intermediate":
                    intermediate_texts.append(text)
                    logger.debug(f"[任务 {task_id}] 中间结果: {text}")
                elif event == "sentence_end":
                    nonlocal final_text
                    final_text = text
                    logger.info(f"[任务 {task_id}] 最终结果: {text}")
                elif event == "error":
                    logger.error(f"[任务 {task_id}] 错误: {text}")
            
            # 运行识别
            for event, text in asr.run_stream(
                audio_iterable=iter(audio_chunks),
                on_result=on_result,
                sample_rate=16000,
                audio_format="pcm"
            ):
                pass
            
            result["end_time"] = time.time()
            result["duration"] = result["end_time"] - result["start_time"]
            result["raw_text"] = final_text
            
            # 后处理
            if final_text:
                result["processed_text"] = postprocess_asr(final_text)
            
            result["success"] = True
            logger.info(f"[任务 {task_id}] 完成，耗时 {result['duration']:.2f}秒")
            
        except Exception as e:
            result["end_time"] = time.time()
            result["duration"] = result["end_time"] - result["start_time"]
            result["error"] = str(e)
            result["success"] = False
            logger.error(f"[任务 {task_id}] 失败: {e}")
        
        return result
    
    def run_concurrent_test(self) -> List[Dict[str, Any]]:
        """
        运行并发测试
        
        Returns:
            所有任务的结果列表
        """
        logger.info(f"=" * 80)
        logger.info(f"开始并发测试: {self.num_concurrent} 个并发任务")
        logger.info(f"音频文件: {self.audio_file}")
        logger.info(f"=" * 80)
        
        # 加载音频
        audio_chunks = self.load_audio_chunks()
        
        # 使用线程池执行并发任务
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_concurrent) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.single_asr_task, i, audio_chunks): i 
                for i in range(1, self.num_concurrent + 1)
            }
            
            # 收集结果
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    with self.lock:
                        self.results.append(result)
                except Exception as e:
                    logger.error(f"任务 {task_id} 异常: {e}")
        
        total_time = time.time() - start_time
        
        # 打印测试报告
        self.print_report(total_time)
        
        return self.results
    
    def print_report(self, total_time: float):
        """
        打印测试报告
        
        Args:
            total_time: 总耗时
        """
        logger.info(f"\n" + "=" * 80)
        logger.info(f"测试报告")
        logger.info(f"=" * 80)
        
        # 统计
        success_count = sum(1 for r in self.results if r["success"])
        fail_count = len(self.results) - success_count
        
        logger.info(f"总任务数: {len(self.results)}")
        logger.info(f"成功: {success_count}")
        logger.info(f"失败: {fail_count}")
        logger.info(f"成功率: {success_count / len(self.results) * 100:.2f}%")
        logger.info(f"总耗时: {total_time:.2f}秒")
        
        # 成功任务的统计
        if success_count > 0:
            success_results = [r for r in self.results if r["success"]]
            durations = [r["duration"] for r in success_results]
            
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            logger.info(f"\n任务耗时统计:")
            logger.info(f"  平均: {avg_duration:.2f}秒")
            logger.info(f"  最短: {min_duration:.2f}秒")
            logger.info(f"  最长: {max_duration:.2f}秒")
        
        # 详细结果
        logger.info(f"\n详细结果:")
        logger.info(f"-" * 80)
        
        for result in sorted(self.results, key=lambda x: x["task_id"]):
            status = "✓" if result["success"] else "✗"
            logger.info(f"[{status}] 任务 {result['task_id']:2d} | "
                       f"耗时: {result['duration']:6.2f}秒 | "
                       f"事件数: {len(result['events']):3d}")
            
            if result["success"]:
                logger.info(f"    原始: {result['raw_text'][:50]}...")
                logger.info(f"    处理: {result['processed_text'][:50]}...")
            else:
                logger.info(f"    错误: {result['error']}")
            logger.info(f"-" * 80)
        
        # 失败详情
        if fail_count > 0:
            logger.info(f"\n失败任务详情:")
            for result in self.results:
                if not result["success"]:
                    logger.error(f"任务 {result['task_id']}: {result['error']}")


def test_concurrent_asr(audio_file: str = None, num_concurrent: int = 10):
    """
    测试并发ASR
    
    Args:
        audio_file: 音频文件路径，默认使用项目中的测试音频
        num_concurrent: 并发数量
    """
    # 默认音频文件
    if audio_file is None:
        audio_file = str(project_root / "data" / "audio" / "output.wav")
    
    # 检查文件是否存在
    if not os.path.exists(audio_file):
        logger.error(f"音频文件不存在: {audio_file}")
        logger.info(f"请提供有效的音频文件路径")
        return
    
    # 创建测试器并运行
    tester = ConcurrentASRTester(audio_file, num_concurrent)
    results = tester.run_concurrent_test()
    
    return results


def test_stress(audio_file: str = None):
    """
    压力测试：逐步增加并发数
    
    Args:
        audio_file: 音频文件路径
    """
    if audio_file is None:
        audio_file = str(project_root / "data" / "audio" / "output.wav")
    
    if not os.path.exists(audio_file):
        logger.error(f"音频文件不存在: {audio_file}")
        return
    
    logger.info(f"\n" + "=" * 80)
    logger.info(f"压力测试：逐步增加并发数")
    logger.info(f"=" * 80)
    
    concurrent_levels = [1, 2, 5, 10, 15, 20]
    
    for level in concurrent_levels:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"测试并发数: {level}")
        logger.info(f"{'=' * 80}")
        
        tester = ConcurrentASRTester(audio_file, level)
        tester.run_concurrent_test()
        
        # 等待一段时间再进行下一轮测试
        time.sleep(2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RealtimeFunASR 并发测试")
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="音频文件路径 (默认: data/audio/output.wav)"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="并发数量 (默认: 10)"
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="运行压力测试（逐步增加并发数）"
    )
    
    args = parser.parse_args()
    
    if args.stress:
        # 压力测试
        test_stress(args.audio)
    else:
        # 单次并发测试
        test_concurrent_asr(args.audio, args.concurrent)

