"""
Asynchronous OpenAI API request processors for handling multiple concurrent requests efficiently.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import time
from openai import AsyncOpenAI, OpenAI
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OpenAIRequest:
    """Data class for OpenAI API requests."""

    messages: List[Dict[str, str]]
    model: str = "gpt-4o-mini-2024-07-18"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    request_id: Optional[str] = None


@dataclass
class OpenAIResponse:
    """Data class for OpenAI API responses."""

    content: str
    request_id: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class BaseOpenaiProcessor:
    """Base class for OpenAI API processing with synchronous requests."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI client."""
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def process_single_request(self, request: OpenAIRequest) -> OpenAIResponse:
        """Process a single OpenAI request synchronously."""
        try:
            response = self.client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            content = response.choices[0].message.content.strip()
            usage = (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            )

            return OpenAIResponse(
                content=content,
                request_id=request.request_id,
                success=True,
                usage=usage,
            )

        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {str(e)}")
            return OpenAIResponse(
                content="", request_id=request.request_id, success=False, error=str(e)
            )

    def process_requests(self, requests: List[OpenAIRequest]) -> List[OpenAIResponse]:
        """Process multiple requests synchronously."""
        responses = []
        for request in requests:
            response = self.process_single_request(request)
            responses.append(response)
        return responses


class AsyncOpenaiProcessor:
    """Asynchronous OpenAI API processor for handling multiple concurrent requests."""

    def __init__(self, api_key: Optional[str] = None, max_concurrent: int = 10):
        """
        Initialize with AsyncOpenAI client.

        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
            max_concurrent: Maximum number of concurrent requests
        """
        self.client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    async def process_single_request(self, request: OpenAIRequest) -> OpenAIResponse:
        """Process a single OpenAI request asynchronously with rate limiting."""
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=request.model,
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

                content = response.choices[0].message.content.strip()
                usage = (
                    {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    if response.usage
                    else None
                )

                return OpenAIResponse(
                    content=content,
                    request_id=request.request_id,
                    success=True,
                    usage=usage,
                )

            except Exception as e:
                logger.error(
                    f"Error processing async request {request.request_id}: {str(e)}"
                )
                return OpenAIResponse(
                    content="",
                    request_id=request.request_id,
                    success=False,
                    error=str(e),
                )

    async def process_requests(
        self, requests: List[OpenAIRequest]
    ) -> List[OpenAIResponse]:
        """Process multiple requests asynchronously."""
        tasks = [self.process_single_request(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(
                    f"Exception in request {requests[i].request_id}: {str(response)}"
                )
                processed_responses.append(
                    OpenAIResponse(
                        content="",
                        request_id=requests[i].request_id,
                        success=False,
                        error=str(response),
                    )
                )
            else:
                processed_responses.append(response)

        return processed_responses

    def run_batch(self, requests: List[OpenAIRequest]) -> List[OpenAIResponse]:
        """
        Convenience method to run batch processing in a synchronous context.
        Creates and runs an event loop if none exists.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we need to use a different approach
            raise RuntimeError(
                "Already in async context - use process_requests() directly"
            )
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(self.process_requests(requests))


# Utility functions for common use cases


def create_classification_request(
    text: str, request_id: str = None, model: str = "gpt-4o-mini-2024-07-18"
) -> OpenAIRequest:
    """Create a request for text classification."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful text classification assistant.",
        },
        {"role": "user", "content": f"Classify this text: {text}"},
    ]
    return OpenAIRequest(
        messages=messages,
        model=model,
        request_id=request_id or f"classify_{int(time.time())}",
    )


def create_spam_detection_request(
    text: str, request_id: str = None, model: str = "gpt-4o-mini-2024-07-18"
) -> OpenAIRequest:
    """Create a request for spam detection."""
    messages = [
        {"role": "system", "content": "You are a spam detection assistant."},
        {"role": "user", "content": f"Analyze if this text is spam: {text}"},
    ]
    return OpenAIRequest(
        messages=messages,
        model=model,
        request_id=request_id or f"spam_{int(time.time())}",
    )


def create_answer_generation_request(
    query: str,
    context: str,
    request_id: str = None,
    model: str = "gpt-4o-mini-2024-07-18",
) -> OpenAIRequest:
    """Create a request for answer generation."""
    messages = [
        {"role": "system", "content": "You are a helpful customer service assistant."},
        {
            "role": "user",
            "content": f"Query: {query}\n\nContext: {context}\n\nPlease provide a helpful answer.",
        },
    ]
    return OpenAIRequest(
        messages=messages,
        model=model,
        request_id=request_id or f"answer_{int(time.time())}",
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the async processor
    async def test_async():
        processor = AsyncOpenaiProcessor(max_concurrent=5)

        # Create test requests
        requests = [
            OpenAIRequest(
                messages=[{"role": "user", "content": f"Count to {i}"}],
                request_id=f"test_{i}",
            )
            for i in range(1, 4)
        ]

        start_time = time.time()
        responses = await processor.process_requests(requests)
        end_time = time.time()

        print(
            f"Processed {len(responses)} requests in {end_time - start_time:.2f} seconds"
        )
        for response in responses:
            print(
                f"Request {response.request_id}: {'Success' if response.success else 'Failed'}"
            )
            if response.success:
                print(f"  Content: {response.content[:100]}...")
            else:
                print(f"  Error: {response.error}")

    # Run the test
    print("Testing AsyncOpenaiProcessor...")
    asyncio.run(test_async())
