import signal
import functools
import os
import errno
import time

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

@timeout(6)
def long_running_function1():
    time.sleep(5)
    print("success with setting 6 seconds ")

@timeout(5)
def long_running_function2():
    time.sleep(5)
    print("success with setting 5 seconds ")

@timeout(3, os.strerror(errno.ETIMEDOUT))
def long_running_function3():
    time.sleep(5)
    print("success with setting 3 seconds ")

if __name__ == '__main__':
    try:
        long_running_function1()
    except Exception as e:
        print(e)
    
    try:
        long_running_function2()
    except Exception as e:
        print(e)

    try:
        long_running_function3()
    except Exception as e:
        print(e)