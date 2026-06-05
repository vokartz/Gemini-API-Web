import unittest

from gemini_webapi.server.app import (
    ChatCompletionRequest,
    ChatMessage,
    _append_tool_instructions,
    _messages_to_prompt,
    _tool_calls_from_output_text,
)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class ServerToolCallTests(unittest.TestCase):
    def test_messages_include_multimodal_image_urls(self):
        prompt = _messages_to_prompt(
            [
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "请分析这张图"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/cat.png"},
                        },
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/dog.png",
                        },
                    ],
                )
            ]
        )

        self.assertIn("请分析这张图", prompt)
        self.assertIn("Image URL: https://example.com/cat.png", prompt)
        self.assertIn("Image URL: https://example.com/dog.png", prompt)

    def test_chat_request_accepts_openai_tools(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": "gemini",
                "messages": [{"role": "user", "content": "北京天气怎么样？"}],
                "tools": [WEATHER_TOOL],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
            }
        )

        self.assertEqual(request.tools[0].function.name, "get_weather")
        self.assertIn("get_weather", _append_tool_instructions("User: hi", request))

    def test_messages_include_tool_history(self):
        prompt = _messages_to_prompt(
            [
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"北京"}',
                            },
                        }
                    ],
                ),
                ChatMessage(
                    role="tool",
                    tool_call_id="call_1",
                    content='{"temperature":"22C"}',
                ),
            ]
        )

        self.assertIn("Assistant tool calls:", prompt)
        self.assertIn("Tool result (call_1):", prompt)

    def test_parses_openai_tool_calls_from_model_json(self):
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "天气"}],
                "tools": [WEATHER_TOOL],
            }
        )
        calls = _tool_calls_from_output_text(
            '{"tool_calls":[{"name":"get_weather","arguments":{"city":"北京"}}]}',
            request.tools,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(calls[0]["function"]["arguments"], '{"city":"北京"}')

    def test_ignores_unknown_tool_names(self):
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "天气"}],
                "tools": [WEATHER_TOOL],
            }
        )
        calls = _tool_calls_from_output_text(
            '{"tool_calls":[{"name":"delete_everything","arguments":{}}]}',
            request.tools,
        )

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
