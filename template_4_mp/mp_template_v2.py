from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time


def process_sample(arg):
    index, b = arg
    result = b*b
    return {
        index:result
    }

if __name__ == "__main__":
    data = list(range(100, 1000))
    args = []
    for i in range(len(data)):
        args.append((i, data[i]))

    ## 单线程翻遍调试
    start = time.time()
    for i in tqdm(range(len(args))):
        process_sample(args[i])
    print(f"single:{time.time()-start:.3f}")

    num_workers = min(10, cpu_count())

    start = time.time()
    with Pool(processes=num_workers) as pool:
        try:
            results = pool.map(process_sample, args) #list
        except Exception as e:
            print(f"Processing error: {e}")
        finally:
            pool.close()
            pool.join()
    print(f"mp:{time.time()-start:.3f}")