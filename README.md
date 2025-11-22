# Caller

This simple module provides a uniform interface for making API calls to Openrouter, while automatically handling the following (with easy configuration):

- Limiting max parallel requests
- Sensible retries
- Response caching

See `demo.py` for basic usage.

**WARNING**: The caching logic uses a single DB lock and will slow you down. If you have a lot of async read/writes, you should disable caching!

## Acknowledgements

This module was heavily modified from some code initially shared with me by Adam Karvonen, who in turn got it from James Chua.

## To do

- [ ] Ensure correct prefill behavior
- [x] Add OpenAI native API support
- [x] Add Anthropic native API support
- [ ] Implement proper async DB r/w
- [ ] Add image support
- [ ] Add streaming response support
