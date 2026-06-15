from pathlib import Path

from pier.agents.factory import AgentFactory
from pier.agents.installed.pi import Pi
from pier.models.agent.name import AgentName


def test_pi_is_registered(tmp_path: Path):
    agent = AgentFactory.create_agent_from_name(
        AgentName.PI,
        logs_dir=tmp_path,
        model_name="openai/gpt-5.4-mini",
    )

    assert isinstance(agent, Pi)
    assert agent.name() == "pi"


def test_pi_install_spec_uses_npm_package(tmp_path: Path):
    agent = Pi(logs_dir=tmp_path, model_name="openai/gpt-5.4-mini")

    spec = agent.install_spec()

    assert spec.agent_name == "pi"
    assert "npm install -g --ignore-scripts @mariozechner/pi-coding-agent" in spec.steps[1].run
    assert spec.verification_command is not None
    assert "pi --version" in spec.verification_command


def test_pi_reports_openai_domains(tmp_path: Path):
    agent = Pi(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.4-mini",
        extra_env={"OPENAI_BASE_URL": "https://gateway.example.com/v1"},
    )

    domains = set(agent.network_allowlist().domains)

    assert {"api.openai.com", "gateway.example.com"} <= domains


def test_pi_uses_openai_provider_for_unprefixed_model(tmp_path: Path):
    agent = Pi(logs_dir=tmp_path, model_name="gpt-5.4-mini")

    assert agent._model_args() == "--provider openai --model gpt-5.4-mini"


def test_pi_converts_json_events_to_atif(tmp_path: Path):
    agent = Pi(logs_dir=tmp_path, model_name="openai/gpt-5.4-mini")
    events = [
        {
            "type": "assistant_message",
            "session_id": "session-1",
            "message": {"content": [{"type": "text", "text": "I fixed it."}]},
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cost_usd": 0.01,
            },
            "timestamp_ms": 1000,
        }
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.agent.name == "pi"
    assert trajectory.session_id == "session-1"
    assert trajectory.steps[0].message == "I fixed it."
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_completion_tokens == 20


def test_pi_converts_message_end_usage_to_metrics(tmp_path: Path):
    agent = Pi(logs_dir=tmp_path, model_name="openai/gpt-5.4-mini")
    events = [
        {"type": "session", "id": "session-1"},
        {
            "type": "message_end",
            "message": {
                "id": "msg-1",
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 20,
                    "cacheReadTokens": 30,
                    "cacheWriteTokens": 40,
                    "cost": 0.05,
                },
            },
        },
        {
            "type": "turn_end",
            "message": {
                "id": "msg-1",
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 20,
                    "cacheReadTokens": 30,
                    "cost": 0.05,
                },
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert len(trajectory.steps) == 1
    assert trajectory.session_id == "session-1"
    assert trajectory.steps[0].metrics is not None
    assert trajectory.steps[0].metrics.prompt_tokens == 130
    assert trajectory.steps[0].metrics.completion_tokens == 20
    assert trajectory.steps[0].metrics.cached_tokens == 30
    assert trajectory.steps[0].metrics.cost_usd == 0.05
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 130
    assert trajectory.final_metrics.total_completion_tokens == 20
    assert trajectory.final_metrics.total_cached_tokens == 30
    assert trajectory.final_metrics.total_cost_usd == 0.05
