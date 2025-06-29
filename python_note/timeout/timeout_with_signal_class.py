import time
import signal

class Timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

if __name__ == '__main__':
    try:
        with Timeout(seconds=6):
            time.sleep(5)
        print("success with setting 6 seconds ")
    except TimeoutError as e:
        print(e)

    try:
        with Timeout(seconds=5):
            time.sleep(5)
        print("success with setting 5 seconds ")
    except TimeoutError as e:
        print(e)

    try:
        with Timeout(seconds=3):
            time.sleep(5)
        print("success with setting 3 seconds ")
    except TimeoutError as e:
        print(e)