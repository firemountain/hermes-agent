import asyncio
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def test_direct_conversation_is_one_tool_free_provider_stream(monkeypatch):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        delta = SimpleNamespace(content="hello")
        yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call)
    monkeypatch.setattr("agent.auxiliary_client._read_main_provider", lambda: "configured-provider")
    monkeypatch.setattr("agent.auxiliary_client._read_main_model", lambda: "configured-model")

    async def scenario():
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
            "key": "a-secure-api-server-key-123456789", "direct_conversation": True,
        }))
        app = web.Application()
        app.router.add_post("/api/direct-conversation/stream", adapter._handle_direct_conversation_stream)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/api/direct-conversation/stream", headers={
                "Authorization": "Bearer a-secure-api-server-key-123456789",
            }, json={"instructions": "Stable voice", "message": "Hi", "history": []})
            assert response.status == 200
            text = await response.text()
            assert 'event: delta' in text and '"text": "hello"' in text

    asyncio.run(scenario())
    assert len(calls) == 1
    assert calls[0]["tools"] is None
    assert calls[0]["provider"] == "configured-provider"
    assert calls[0]["model"] == "configured-model"
    assert calls[0]["stream"] is True


def test_direct_conversation_accepts_one_shot_completion_from_stream_request(monkeypatch):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="one-shot"))])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call)
    monkeypatch.setattr("agent.auxiliary_client._read_main_provider", lambda: "configured-provider")
    monkeypatch.setattr("agent.auxiliary_client._read_main_model", lambda: "configured-model")

    async def scenario():
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
            "key": "a-secure-api-server-key-123456789", "direct_conversation": True,
        }))
        app = web.Application()
        app.router.add_post("/api/direct-conversation/stream", adapter._handle_direct_conversation_stream)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/api/direct-conversation/stream", headers={
                "Authorization": "Bearer a-secure-api-server-key-123456789",
            }, json={"instructions": "Stable voice", "message": "Hi", "history": []})
            assert response.status == 200
            text = await response.text()
            assert 'event: delta' in text and '"text": "one-shot"' in text
            assert 'event: done' in text and 'event: error' not in text

    asyncio.run(scenario())
    assert len(calls) == 1
    assert calls[0]["tools"] is None
    assert calls[0]["stream"] is True


def test_direct_history_is_bounded_alternating_and_route_is_config_gated():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"direct_conversation": False}))
    assert ("POST", "/api/direct-conversation/stream", adapter._handle_direct_conversation_stream) in adapter._http_route_table()
    messages = adapter._direct_messages({
        "instructions": "Stable", "message": "Now", "evidence": "Citation",
        "history": [{"role": "assistant", "content": "ignored"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
    })
    assert [item["role"] for item in messages] == ["system", "user", "assistant", "user"]
    assert "Citation" in messages[-1]["content"]


def test_health_reports_packaged_and_actual_direct_feature_state():
    async def scenario():
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"direct_conversation": False}))
        request = SimpleNamespace()
        response = await adapter._handle_health(request)
        body = __import__("json").loads(response.text)
        assert body["capabilities"]["direct_conversation"] == {"packaged": True, "enabled": False}
    asyncio.run(scenario())


def test_provider_failure_after_headers_is_typed_terminal_error(monkeypatch):
    def failing_call(**_kwargs):
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))])
        raise RuntimeError("secret provider detail")
    monkeypatch.setattr("agent.auxiliary_client.call_llm", failing_call)
    monkeypatch.setattr("agent.auxiliary_client._read_main_provider", lambda: "provider")
    monkeypatch.setattr("agent.auxiliary_client._read_main_model", lambda: "model")
    async def scenario():
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "a-secure-api-server-key-123456789", "direct_conversation": True}))
        app = web.Application(); app.router.add_post("/api/direct-conversation/stream", adapter._handle_direct_conversation_stream)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/api/direct-conversation/stream", headers={"Authorization": "Bearer a-secure-api-server-key-123456789"}, json={"instructions": "stable", "message": "hello"})
            text = await response.text()
            assert 'event: error' in text and '"code": "provider_error"' in text
            assert 'event: done' not in text and "secret provider detail" not in text
    asyncio.run(scenario())
