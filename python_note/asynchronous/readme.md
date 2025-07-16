## 介绍
异步编程(Asynchronous )是一种并发方法，允许程序同时执行多个任务。在Python中，这对于提高**I/O**结合和**高延迟应用**的性能尤为重要。

## 模板
```python
# Synchronous Function
def sync_function():
    # Perform tasks sequentially
    pass

# Asynchronous Function
async def async_function():
    # Perform tasks concurrently
    pass
```

## 在Python中编写异步代码
```python
import asyncio

## Use async def to define a coroutine.
async def main():
    print('Hello')
    ## await is used to call other asynchronous functions.
    await asyncio.sleep(1)
    print('World')

## Use asyncio.run() to run the main coroutine.
asyncio.run(main())
```

## Handling I/O-bound and High-latency Operations
Asynchronous programming excels in I/O-bound and high-latency scenarios.

Code Example: Asynchronous HTTP Request
```python
import aiohttp
import asyncio

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, 'http://python.org')
        print(html)

asyncio.run(main())
```

## Advanced Asynchronous Programming Techniques
```python
import asyncio

async def task(name, seconds):
    print(f'Task {name} started')
    await asyncio.sleep(seconds)
    print(f'Task {name} completed')

async def main():
    ## Utilize asyncio.gather() to run multiple coroutines concurrently.
    await asyncio.gather(
        task('A', 2),
        task('B', 3),
    )

asyncio.run(main())
```


## ref
https://codefinity.com/blog/Asynchronous-Programming-in-Python