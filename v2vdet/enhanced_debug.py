import sys
import ipdb
import signal
import os
import threading
from functools import partial
from typing import Any, Callable, Optional, Type

class EnhancedDebugger:
    def __init__(self):
        self.pid = os.getpid()
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._active = False
        self._debug_event = threading.Event()
        self._current_frame = None
    
    def setup(self) -> None:
        """設置整合式除錯環境"""
        if self._active:
            print("[!] 除錯器已經在運行中")
            return
            
        try:
            # 設置信號處理器
            signal.signal(signal.SIGINT, self._handle_interrupt)
            signal.signal(signal.SIGUSR1, self._handle_debug_signal)
            self._active = True
            
            # 啟動除錯監控執行緒
            self._start_debug_monitor()
            
            print(f"""
=== 增強型除錯環境已啟動 (PID: {self.pid}) ===
在 VSCode 斷點停下時，可以：
1. 按 Ctrl+C 進入 ipdb 互動式除錯
   (要按兩次 Ctrl+C：第一次進入 ipdb，第二次退出程式)
2. 在另一個終端機執行以觸發 ipdb：
   kill -SIGUSR1 {self.pid}

ipdb 常用指令：
   n(ext)      : 下一行
   s(tep)      : 進入函式
   c(ontinue)  : 繼續執行
   p 變數      : 印出變數
   pp 變數     : 格式化印出
   w(here)     : 顯示堆疊
   l           : 顯示程式碼
   q           : 退出除錯
""")
        except Exception as e:
            print(f"[!] 設置除錯環境時發生錯誤: {e}")
            self._restore_signals()

    def _start_debug_monitor(self) -> None:
        """啟動除錯監控執行緒"""
        def monitor():
            while self._active:
                if self._debug_event.wait(0.1):  # 等待除錯事件
                    self._debug_event.clear()
                    if self._current_frame:
                        try:
                            print("\n[!] 啟動 ipdb 除錯會話...")
                            # 在主執行緒中執行 ipdb
                            sys.settrace(None)  # 清除現有的追蹤
                            ipdb.set_trace(self._current_frame)
                        except Exception as e:
                            print(f"[!] ipdb 啟動失敗: {e}")
                        finally:
                            self._current_frame = None

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    def _trigger_debug(self, frame: Any) -> None:
        """觸發除錯會話"""
        self._current_frame = frame
        self._debug_event.set()
    
    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """處理 Ctrl+C"""
        print("\n[!] 收到中斷信號，準備進入除錯模式")
        self._trigger_debug(frame)
        
    def _handle_debug_signal(self, signum: int, frame: Any) -> None:
        """處理 SIGUSR1 信號"""
        print("\n[!] 收到除錯信號，準備進入除錯模式")
        self._trigger_debug(frame)

    def _restore_signals(self) -> None:
        """還原原始信號處理器"""
        try:
            signal.signal(signal.SIGINT, self._original_sigint)
            signal.signal(signal.SIGUSR1, signal.default_int_handler)
            self._active = False
            if hasattr(self, '_monitor_thread'):
                self._debug_event.set()  # 喚醒監控執行緒
                self._monitor_thread.join(timeout=1.0)
        except Exception as e:
            print(f"[!] 還原信號處理器時發生錯誤: {e}")
    
    def __call__(self, func: Callable) -> Callable:
        """裝飾器功能"""
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"\n[!] 錯誤發生: {str(e)}")
                print("[i] 進入後期除錯模式")
                try:
                    ipdb.post_mortem()
                except Exception as debug_error:
                    print(f"[!] 後期除錯啟動失敗: {debug_error}")
                raise
        return wrapper
    
    def debug(self) -> None:
        """手動觸發除錯器"""
        try:
            ipdb.set_trace()
        except Exception as e:
            print(f"[!] 手動觸發除錯器失敗: {e}")
    
    def cleanup(self) -> None:
        """清理並還原信號處理器"""
        if self._active:
            self._restore_signals()
            print("[i] 除錯環境已清理")

# 創建全域除錯器實例
debugger = EnhancedDebugger()