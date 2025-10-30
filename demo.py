"""
Example usage of the Caller library.
"""
import asyncio
import time
from contextlib import contextmanager
from caller import Caller, CacheConfig
from caller.types import OpenaiResponse, ChatHistory, InferenceConfig


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
        no_cache_models={"meta-llama/llama-3.1-8b-instruct"},
    )

    # Use async context manager to ensure connections are closed
    async with Caller(cache_config=cache_config) as caller:
        messages = [
            "What is the expected number of times do I have to throw a coin before I first get a sequence of HTH?",
        ]

        # Process in parallel with shared parameters
        responses = await caller.call(
            messages=messages,
            max_parallel=128,
            model="openai/gpt-5-mini",
            desc="Sending prompts",
            disable_cache=True,
            max_tokens=4096,
            reasoning=3000
        )

        print("Reasoning content: ", responses[0].reasoning_content)

        for question, response in zip(messages, responses):
            print(f"Q: {question}")
            print(f"A: {response.first_response}\n")


# Response caching

async def cache_demo():
    """Example showing cache hits and misses."""
    async with Caller() as caller:
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


async def parallel_hit_stress(num_calls: int = 128) -> None:
    """
    Stress test: many parallel calls to the same model.
    """
    tasks = []
    caller = Caller()

    async def make_call():
        async with caller:
            response = await caller.call_one(
                messages="Hello! Can you give me a random joke? Sample a random one from the entire distribution, according to their probabilities.", 
                model="gpt-5-nano",
                reasoning="low",
                max_tokens=2048,
                disable_cache=True,
            )
            print("Got a response")
            return response
    
    for _ in range(num_calls):
        tasks.append(make_call())
    
    return await asyncio.gather(*tasks)


async def test_claude():
    caller = Caller(provider="anthropic", cache_config=CacheConfig(base_path=None))
    response = await caller.call_one(
        messages="Hello! Can you give me 5 jokes? Sample from the full distribution, as well as their probabilities.", 
        model="claude-sonnet-4-5",
        reasoning=2000,
        max_tokens=2048
    )
    print("Usage: ", response.usage, "\n\n")
    print("Reasoning content: ", response.reasoning_content + "\n\n")
    print(response.first_response)

if __name__ == "__main__":
    asyncio.run(test_claude())
