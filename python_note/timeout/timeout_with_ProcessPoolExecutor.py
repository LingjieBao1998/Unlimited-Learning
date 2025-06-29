from concurrent.futures import TimeoutError
from time import sleep
import concurrent

max_numbers = [1000]

def run_loop(max_number):
    sleep(max_number)

def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(max_numbers)) as executor:
        try:
            for future in concurrent.futures.as_completed(executor.map(run_loop, max_numbers, timeout=1), timeout=1):
                print(future.result(timeout=1))
        except concurrent.futures._base.TimeoutError:
            print("This took to long...")
            stop_process_pool(executor)

def stop_process_pool(executor):
    for pid, process in executor._processes.items():
        process.terminate()
    executor.shutdown()

if __name__ == '__main__':
    main()