# debug_attach.py
import multiprocessing
import code
import readline
import rlcompleter
import socket
import time
import debugpy
from debugpy.server import api

class DebugInteractiveConsole:
    def __init__(self):
        self.locals = {}
        self.globals = globals()
        
    def setup(self):
        """設置互動式除錯環境"""
        try:
            # 設置自動補全
            readline.parse_and_bind("tab: complete")
            
            print("正在連接到除錯服務...")
            
            # 等待 debugpy 服務可用
            self._wait_for_debugpy()
            
            # 啟動互動式會話
            code.InteractiveConsole(locals=self.locals).interact(
                banner="""
Debug Interactive Console Ready
你現在可以在這個終端機中進行互動式除錯
當程式在 VSCode 中遇到斷點時，可以在這裡查看和修改變數
輸入 'quit()' 來結束除錯會話

可用的指令：
- 輸入變數名稱來查看值
- 使用 dir() 查看可用的物件
- 執行任何 Python 表達式
"""
            )
            
        except Exception as e:
            print(f"設置失敗: {e}")
            print("請確保主程式已經運行並啟動了 debugpy 服務")
    
    def _wait_for_debugpy(self, timeout=30):
        """等待 debugpy 服務啟動"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 測試連接
                with socket.create_connection(('localhost', 5678), timeout=1):
                    print("成功偵測到除錯服務")
                    return True
            except (socket.timeout, ConnectionRefusedError):
                time.sleep(1)
        raise TimeoutError("等待除錯服務超時")

if __name__ == '__main__':
    console = DebugInteractiveConsole()
    console.setup()