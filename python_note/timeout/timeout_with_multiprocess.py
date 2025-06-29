import multiprocessing
import time

def long_running_function():
    time.sleep(10)  # 模拟一个耗时10秒的任务
    print("success in `long_running_function`")

def long_running_function2():
    time.sleep(1)  # 模拟一个耗时10秒的任务
    print("success in `long_running_function2`")

if __name__ == '__main__':
    p = multiprocessing.Process(target=long_running_function)
    p.start()         # 启动子进程
    p.join(3)         # 等待子进程最多3秒
    if p.is_alive():  # 如果子进程还在运行
        print("timeout, terminated!")
        p.terminate() # 强制终止子进程
        p.join()      # 等待子进程真正结束（清理资源）
    
    p = multiprocessing.Process(target=long_running_function2)
    p.start()         # 启动子进程
    p.join(3)         # 等待子进程最多3秒
    if p.is_alive():  # 如果子进程还在运行
        print("timeout, terminated!")
        p.terminate() # 强制终止子进程
        p.join()      # 等待子进程真正结束（清理资源）