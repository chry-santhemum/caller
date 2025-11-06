# Caller

This simple module provides a uniform interface for making API calls to Openrouter, while automatically handling the following (with easy configuration):

- Limiting max parallel requests
- Sensible retries
- Response caching

See `demo.py` for basic usage.

## Acknowledgements

This module was heavily modified from some code initially shared with me by Adam Karvonen, who in turn got it from James Chua.

## To do

- Add back OpenAI and Anthropic native support
- Add image support
- Add streaming response support
