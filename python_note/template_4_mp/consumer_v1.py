from multiprocessing import Queue, Process


def process_data(index, data):
    # print(index, data)
    pass

def worker(queue):
    print("initiate queue")
    while True:
        ## get data
        index, data = queue.get()
        if data is None:
            break
        process_data(index, data)


if __name__ == "__main__":
    queue = Queue()

    # Start worker processes
    processes = []
    num_workers = 3
    for _ in range(num_workers):
        p = Process(target=worker, args=(queue,))
        p.start()
        processes.append(p)
    
    ## add data
    for index, i in enumerate(range(100, 1000)):
        queue.put((index, i))
    
    for _ in range(num_workers):
        queue.put((None, None))

    for p in processes:
        p.join()
    
    
    
