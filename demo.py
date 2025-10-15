"""
Example usage of the Caller library.
"""
import asyncio
import time
from contextlib import contextmanager
from caller import Caller


@contextmanager
def timer(description: str = "Operation"):
    """Context manager to measure wallclock time."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  [{description}] took {elapsed:.3f}s")


async def basic_usage():
    # Initialize the caller
    caller = Caller()

    # One message
    response = await caller.call_one(
        messages="What is the capital of France?",
        model="anthropic/claude-3.5-haiku",
        max_tokens=50,
    )

    print(f"Answer: {response.first_response}")
    print(f"Tokens used: {response.usage}")


    # Multiple messages with shared parameters
    questions = [
        "What is the capital of Japan?",
        "What is the capital of Germany?",
        "What is the capital of Brazil?",
    ]

    print(f"Processing {len(questions)} questions in parallel...\n")

    # Process in parallel with shared parameters
    responses = await caller.call(
        questions,
        model="anthropic/claude-3.5-haiku",
        max_tokens=30,
        max_parallel=3
    )

    for question, response in zip(questions, responses):
        print(f"Q: {question}")
        print(f"A: {response.first_response}\n")


# Using different providers

async def different_providers():
    caller = Caller()

    # Default: all models use OpenRouter
    response = await caller.call_one(
        messages="Say hello",
        model="anthropic/claude-3.5-haiku",  # via OpenRouter
        max_tokens=10,
    )
    print(f"   {response.first_response}")

    # Explicitly use Anthropic Direct API
    response = await caller.call_one(
        messages="Say hello",
        model="claude-sonnet-4-5-20250929",
        provider="anthropic",  # Explicit override
        max_tokens=10,
    )
    print(f"   {response.first_response}")

    # Explicitly use OpenAI Direct API
    response = await caller.call_one(
        messages="Say hello",
        model="gpt-5-mini",
        provider="openai",  # Explicit override
        max_tokens=10,
    )
    print(f"   {response.first_response}")



# Response caching

async def cache_demo():
    caller = Caller()

    message = "What is 2+2?"
    model = "anthropic/claude-sonnet-4-5-20250929"

    print("First call (hits API)...")
    with timer("API call"):
        response1 = await caller.call_one(message, model=model, max_tokens=10)
    print(f"  Response: {response1.first_response}")

    print("\nSecond call (hits cache)...")
    with timer("Cache hit"):
        response2 = await caller.call_one(message, model=model, max_tokens=10)
    print(f"  Response: {response2.first_response}")
    print(f"  Same as first? {response1.first_response == response2.first_response}")

    print("\nThird call with different parameters (hits API)...")
    with timer("API call"):
        response3 = await caller.call_one(message, model=model, max_tokens=10, temperature=0.9)
    print(f"  Response: {response3.first_response}")



if __name__ == "__main__":
    asyncio.run(basic_usage())
