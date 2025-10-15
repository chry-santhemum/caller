"""
Example usage of the Caller library.
"""
import asyncio
import time
from contextlib import contextmanager
from caller import Caller, CacheConfig


@contextmanager
def timer(description: str = "Operation"):
    """Context manager to measure wallclock time."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  [{description}] took {elapsed:.3f}s")


cache_config = CacheConfig(
    no_cache_models={"meta-llama/llama-3.1-8b-instruct"},
)

# Initialize the caller

caller = Caller(cache_config=cache_config)


async def basic_usage():
    messages = [
        "What is the capital of Japan? Respond with a poem.",
        "What is the capital of Germany? Respond with a poem.",
        "What is the capital of Brazil? Respond with a poem.",
    ]

    # Process in parallel with shared parameters
    responses = await caller.call(
        messages=messages,
        max_parallel=128,
        model="meta-llama/llama-3.1-8b-instruct",
        desc="Sending prompts",
        max_tokens=128
    )

    print(f"Responses: {responses}")

    for question, response in zip(messages, responses):
        print(f"Q: {question}")
        print(f"A: {response.first_response}\n")


# Response caching

async def cache_demo():

    message = "Who is Yo Mama?"
    model = "meta-llama/llama-3.1-70b-instruct"

    print("First call...")
    with timer("API call"):
        response1 = await caller.call_one(messages=message, model=model, max_tokens=128)
    print(f"  Response: {response1.first_response}")

    print("\nSecond call...")
    with timer("Cache hit"):
        response2 = await caller.call_one(messages=message, model=model, max_tokens=128)
    print(f"  Response: {response2.first_response}")

    print("\nThird call...")
    with timer("API call"):
        response3 = await caller.call_one(messages=message, model=model, max_tokens=128, temperature=0.9)
    print(f"  Response: {response3.first_response}")



if __name__ == "__main__":
    asyncio.run(cache_demo())
