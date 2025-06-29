import func_timeout
from func_timeout import func_set_timeout

@func_set_timeout(6)
def long_running_function1():
    time.sleep(5)
    print("success with setting 6 seconds ")

@func_set_timeout(5)
def long_running_function2():
    time.sleep(5)
    print("success with setting 5 seconds ")

@func_set_timeout(3)
def long_running_function3():
    time.sleep(5)
    print("success with setting 3 seconds ")

if __name__ == '__main__':
    try:
        long_running_function1()
    except func_timeout.FunctionTimedOut as e:
        print(e)
    except Exception as e:
        print(e)
    
    try:
        long_running_function2()
    except func_timeout.FunctionTimedOut as e:
        print(e)
    except Exception as e:
        print(e)

    try:
        long_running_function3()
    except func_timeout.FunctionTimedOut as e:
        print(e)
    except Exception as e:
        print(e)