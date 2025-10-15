# Caller

This simple module provides a uniform interface for making API calls to Openrouter, OpenAI and Anthropic, while automatically handling the following:

- Limiting max parallel requests
- Sensible retries
- Rate limiting
- Response caching

See `demo.py` for basic usage.

## Acknowledgements

This module was based on some code initially shared with me by Adam Karvonen, who in turn got it from James Chua.

## To do

Add support for image and tool use.
