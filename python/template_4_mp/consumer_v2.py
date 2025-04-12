from multiprocessing import Queue, Process, Manager
import time

def process_data(index, data):
    """
    模拟数据处理函数。
    这里可以替换为实际的数据处理逻辑。
    """
    # 示例处理：将数据乘以2
    result = data * 2
    # 模拟处理时间
    # time.sleep(0.01)
    return index, result

def worker(queue):
    """
    工作进程函数，从队列中获取数据，处理后将结果放入共享列表。
    """
    print(f"Worker {Process().name} initiated.")
    while True:
        # 从队列中获取数据
        index, data, sharedlist = queue.get()
        if data is None:
            # 接收到终止信号，退出循环
            print(f"Worker {Process().name} received termination signal.")
            break
        # 处理数据
        processed_index, processed_data = process_data(index, data)
        # 将结果放入共享列表
        sharedlist.append((processed_index, processed_data))
    print(f"Worker {Process().name} exiting.")

def main():
    manager = Manager()
    # 创建一个由Manager管理的共享列表
    sharedlist = manager.list()
    
    queue = Queue()

    # 启动工作进程
    num_workers = 3
    processes = []
    for _ in range(num_workers):
        p = Process(target=worker, args=(queue,))
        p.start()
        processes.append(p)
    
    # 添加数据到队列
    total_data = 900  # 从100到999共900个数据项
    for index, i in enumerate(range(100, 1000)):
        queue.put((index, i, sharedlist))
    
    # 发送终止信号给工作进程
    for _ in range(num_workers):
        queue.put((None, None, None))
    
    # 等待所有进程结束
    for p in processes:
        p.join()
    
    # 将共享列表转换为普通列表（可选）
    results = list(sharedlist)
    
    # 打印部分结果以验证
    print(f"Processed {len(results)} items.")
    # 例如，打印前10个结果
    for item in results[:10]:
        print(item)

if __name__ == "__main__":
    main()