import multiprocessing
import time
import random

# 生产者函数
def producer(q, producer_id, num_items):
    for i in range(num_items):
        item = f"P{producer_id}-Item-{i}"
        q.put(item)  # 将数据放入队列
        print(f"生产者 {producer_id} 生产了 {item}")
        time.sleep(random.uniform(0.1, 0.5))  # 模拟生产时间

# 消费者函数
def consumer(q, consumer_id):
    while True:
        try:
            item = q.get(timeout=3)  # 设置超时，防止无限等待
            if item is None:
                print(f"消费者 {consumer_id} 收到终止信号")
                break
            print(f"消费者 {consumer_id} 消费了 {item}")
            time.sleep(random.uniform(0.3, 0.8))  # 模拟消费时间
        except multiprocessing.queues.Empty:
            print(f"消费者 {consumer_id} 因超时退出")
            break

def main():
    num_producers = 2
    num_consumers = 3
    num_items_per_producer = 5
    buffer_size = 5  # 队列大小

    q = multiprocessing.Queue(maxsize=buffer_size)

    # 创建并启动生产者进程
    producers = []
    for i in range(num_producers):
        p = multiprocessing.Process(target=producer, args=(q, i+1, num_items_per_producer))
        producers.append(p)
        p.start()

    # 创建并启动消费者进程
    consumers = []
    for i in range(num_consumers):
        c = multiprocessing.Process(target=consumer, args=(q, i+1))
        consumers.append(c)
        c.start()

    # 等待所有生产者完成
    for p in producers:
        p.join()

    # 发送终止信号给消费者
    for _ in range(num_consumers):
        q.put(None)

    # 等待所有消费者完成
    for c in consumers:
        c.join()

    print("所有生产者和消费者已完成工作。")

if __name__ == "__main__":
    main()