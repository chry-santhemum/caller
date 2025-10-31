import asyncio
import time
from contextlib import contextmanager
from caller import OpenRouterCaller, CacheConfig


@contextmanager
def timer(description: str = "Operation"):
    """Context manager to measure wallclock time."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  [{description}] took {elapsed:.3f}s")


async def basic_usage():
    """Example using async context manager (recommended for automatic cleanup)."""
    cache_config = CacheConfig(
        no_cache_models={"meta-llama/llama-3.1-8b-instruct",},
    )

    # Use async context manager to ensure connections are closed
    async with OpenRouterCaller(cache_config=cache_config) as caller:
        messages = [
            "What is the expected number of times do I have to throw a coin before I first get a sequence of HTH?",
            "Hello! Can you give me 5 jokes? Sample from the full distribution, as well as their probabilities.",
        ]

        # Process in parallel with shared parameters
        responses = await caller.call(
            messages=messages,
            max_parallel=128,
            model="anthropic/claude-sonnet-4.5",
            desc="Sending prompts",
            max_tokens=4096,
            reasoning=3000
        )

        for question, response in zip(messages, responses):
            print(f"Question:\n{question}\n")
            print(f"Response:\n{response.first_response}\n")
            print(f"Reasoning content:\n{response.reasoning_content}\n")


# Response caching

async def cache_demo():
    """Example showing cache hits and misses."""
    async with OpenRouterCaller() as caller:
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
    asyncio.run(basic_usage())
    asyncio.run(cache_demo())
