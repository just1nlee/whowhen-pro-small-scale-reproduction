"""smolagents Model backed by the official Anthropic SDK.

Every API call is appended to `usage_log` (tokens + cost) so pipeline stages
can report exactly what they spent.
"""

import anthropic
from smolagents.models import ChatMessage, MessageRole, Model, get_clean_message_list, tool_role_conversions
from smolagents.monitoring import TokenUsage

# USD per 1M tokens (input, output), Anthropic first-party pricing.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
}


class ClaudeModel(Model):
    def __init__(self, model_id: str = "claude-haiku-4-5", max_tokens: int = 16000, **kwargs):
        if model_id not in PRICES:
            raise ValueError(f"no price listed for {model_id}; add it to PRICES")
        # flatten_messages_as_text: smolagents content blocks -> plain strings
        super().__init__(model_id=model_id, flatten_messages_as_text=True, **kwargs)
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(max_retries=5)
        self.usage_log: list[dict] = []

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        if tools_to_call_from or response_format:
            raise NotImplementedError("ClaudeModel only supports CodeAgent-style text generation")

        msgs = get_clean_message_list(messages, role_conversions=tool_role_conversions, flatten_messages_as_text=True)
        # The Anthropic API takes the system prompt as a separate parameter.
        system = "\n\n".join(m["content"] for m in msgs if m["role"] == MessageRole.SYSTEM)
        chat = [{"role": MessageRole(m["role"]).value, "content": m["content"]}
                for m in msgs if m["role"] != MessageRole.SYSTEM]

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system or anthropic.NOT_GIVEN,
            messages=chat,
            stop_sequences=[s for s in (stop_sequences or []) if s.strip()] or anthropic.NOT_GIVEN,
            **{**kwargs, **self.kwargs},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{self.model_id} refused (request {response._request_id})")

        text = "".join(b.text for b in response.content if b.type == "text")
        self._log(response)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text,
            raw=response,
            token_usage=TokenUsage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens),
        )

    def _log(self, response):
        price_in, price_out = PRICES[self.model_id]
        u = response.usage
        self.usage_log.append({
            "model": self.model_id,
            "request_id": response._request_id,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "stop_reason": response.stop_reason,
            "cost_usd": (u.input_tokens * price_in + u.output_tokens * price_out) / 1e6,
        })

    def total_cost(self) -> float:
        return sum(c["cost_usd"] for c in self.usage_log)
