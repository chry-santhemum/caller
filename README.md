# Caller

This simple module provides a uniform interface for making API calls to Openrouter, OpenAI and Anthropic, while handling the following automatically:

- Limiting max parallel requests
- Sensible retries
- Rate limiting
- Response caching

See `demo.py` for basic usage.

**Acknowledgements:** This module was based on some code shared to me by Adam Karvonen, who in turn got it from James Chua.
