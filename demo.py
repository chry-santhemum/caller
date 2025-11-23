import asyncio
import time
import random
import logging
from contextlib import contextmanager
from caller import AutoCaller, CacheConfig

cache_config = CacheConfig(
    no_cache_models={
        "meta-llama/llama-3.1-8b-instruct",
    },
)

@contextmanager
def timer(description: str = "Operation"):
    """Context manager to measure wallclock time."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  [{description}] took {elapsed:.3f}s")


async def basic_usage():
    caller = AutoCaller(dotenv_path="/workspace/rm-bias/.env")

    messages = [
        "What is the expected number of times do I have to throw a coin before I first get a sequence of HTH?",
        "Hello! Can you give me 5 jokes? Sample from the full distribution, as well as their probabilities.",
    ]

    responses = await caller.call(
        messages=messages,
        max_parallel=128,
        # model="anthropic/claude-sonnet-4.5",
        model="openai/gpt-5-mini",
        desc="Sending prompts",
        max_tokens=2048,
        reasoning="low",
        enable_cache=False,
    )

    for question, response in zip(messages, responses):
        if response is None:
            print(f"Question:\n{question}\n")
            print("Response: None\n")
            continue
        print(f"Question:\n{question}\n")
        print(f"Response:\n{response.first_response}\n")
        print(f"Reasoning content:\n{response.reasoning_content}\n")


async def cache_demo():
    caller = OpenRouterCaller()
    message = "Who is Yo Mama?"
    model = "meta-llama/llama-3.1-70b-instruct"

    print("First call...")
    with timer("API call"):
        response1 = await caller.call_one(messages=message, model=model, max_tokens=1024)
    print(f"  Response: {response1.first_response}")

    print("\nSecond call...")
    with timer("Cache hit"):
        response2 = await caller.call_one(messages=message, model=model, max_tokens=1024)
    print(f"  Response: {response2.first_response}")

    print("\nThird call...")
    with timer("API call"):
        response3 = await caller.call_one(
            messages=message, model=model, max_tokens=1024, temperature=0.9
        )
    print(f"  Response: {response3.first_response}")


async def parallel_requests():
    # no cache
    caller = OpenRouterCaller(cache_config=cache_config)
    messages = ["Please give me a chemical element chosen uniformly at random." for _ in range(1024)]
    with timer("No cache"):
        responses = await caller.call(
            messages=messages,
            max_parallel=128,
            model="meta-llama/llama-3.1-8b-instruct",
            desc="Sending prompts",
            max_tokens=128,
            temperature=random.choice([0.6, 0.7, 0.8, 0.9, 1.0])
        )
    for response in responses[:1]:
        print(f"Response: {response.first_response}")

    # with cache
    caller = OpenRouterCaller()
    messages = ["Please give me a chemical element chosen uniformly at random." for _ in range(1024)]
    with timer("With cache"):
        responses = await caller.call(
            messages=messages,
            max_parallel=128,
            model="meta-llama/llama-3.1-8b-instruct",
            desc="Sending prompts",
            max_tokens=128,
            temperature=random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        )
    for response in responses[:1]:
        print(f"Response: {response.first_response}")


if __name__ == "__main__":
    # add logging to a file
    logging.basicConfig(level=logging.INFO, filename="caller.log")

    asyncio.run(basic_usage())
    # asyncio.run(cache_demo())
    # asyncio.run(parallel_requests())
