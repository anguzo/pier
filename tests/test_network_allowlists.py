from pathlib import Path

from pier.agents.installed.codex import Codex
from pier.agents.installed.gemini_cli import GeminiCli
from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.agents.installed.opencode import OpenCode


def domains(agent) -> set[str]:
    return set(agent.network_allowlist().domains)


def test_codex_reports_custom_and_default_openai_domains(tmp_path: Path):
    agent = Codex(
        logs_dir=tmp_path,
        model_name="gpt-5.5",
        extra_env={"OPENAI_BASE_URL": "https://endpoint.respan.ai/api/"},
    )

    assert {"api.openai.com", "endpoint.respan.ai"} <= domains(agent)


def test_codex_reports_config_toml_provider_base_url(tmp_path: Path):
    agent = Codex(
        logs_dir=tmp_path,
        model_name="gpt-5.5",
        config_toml='openai_base_url = "https://gateway.example.com/v1"\n',
    )

    assert {"api.openai.com", "gateway.example.com"} <= domains(agent)


def test_gemini_cli_reports_custom_respan_base(tmp_path: Path):
    agent = GeminiCli(
        logs_dir=tmp_path,
        model_name="gemini/gemini-3-flash-preview",
        extra_env={
            "GOOGLE_GEMINI_BASE_URL": "https://endpoint.respan.ai/api/google/vertexai/",
            "GEMINI_API_BASE": "https://endpoint.respan.ai/api/google/vertexai/v1beta",
        },
    )

    assert {".googleapis.com", "endpoint.respan.ai"} <= domains(agent)


def test_opencode_reports_openrouter_default_domain(tmp_path: Path):
    agent = OpenCode(
        logs_dir=tmp_path,
        model_name="openrouter/moonshotai/kimi-k2.6",
    )

    assert "openrouter.ai" in domains(agent)


def test_opencode_reports_provider_config_base_url(tmp_path: Path):
    agent = OpenCode(
        logs_dir=tmp_path,
        model_name="google/gemini-3.1-pro-preview",
        opencode_config={
            "provider": {
                "google": {
                    "options": {
                        "baseURL": "https://endpoint.respan.ai/api/google/vertexai/v1beta"
                    }
                }
            }
        },
    )

    assert {".googleapis.com", "endpoint.respan.ai"} <= domains(agent)


def test_opencode_ignores_non_url_provider_api_values(tmp_path: Path):
    agent = OpenCode(
        logs_dir=tmp_path,
        model_name="opencode/custom-model",
        opencode_config={"provider": {"opencode": {"api": "openai-compatible"}}},
    )

    assert "openai-compatible" not in domains(agent)


def test_mini_swe_reports_provider_env_base_urls(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="gemini/gemini-3-flash-preview",
        extra_env={
            "GEMINI_API_BASE": "https://endpoint.respan.ai/api/google/vertexai/v1beta"
        },
    )

    assert {".googleapis.com", "endpoint.respan.ai"} <= domains(agent)


def test_mini_swe_reports_config_yaml_base_url(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        config_yaml="""
model:
  model_kwargs:
    api_base: https://gateway.example.com/v1
""",
    )

    assert {"api.openai.com", "gateway.example.com"} <= domains(agent)


def test_mini_swe_install_refreshes_litellm_cost_map_backup(tmp_path: Path):
    agent = MiniSweAgent(logs_dir=tmp_path, model_name="openai/gpt-5.5")

    spec = agent.install_spec()

    assert spec.steps[-1].env == {"LITELLM_LOCAL_MODEL_COST_MAP": "true"}
    assert MiniSweAgent._LITELLM_MODEL_COST_MAP_URL in spec.steps[-1].run
    assert "model_prices_and_context_window_backup.json" in spec.steps[-1].run


def test_mini_swe_cost_limit_zero_is_config_override(tmp_path: Path):
    agent = MiniSweAgent(logs_dir=tmp_path, model_name="openai/gpt-5.5")

    config_flags = agent._build_config_flags()

    assert "-c mini.yaml" in config_flags
    assert "-c agent.cost_limit=0" in config_flags
    assert "--cost-limit" not in agent.build_cli_flags()


def test_mini_swe_openai_uses_responses_model_class(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        reasoning_effort="xhigh",
    )

    config_flags = agent._build_config_flags()

    assert "-c model.model_class=litellm_response" in config_flags
    assert "-c model.model_kwargs.reasoning_effort=xhigh" in config_flags


def test_mini_swe_model_class_can_be_overridden(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        model_class="litellm",
    )

    assert "-c model.model_class=litellm " in agent._build_config_flags()


def test_mini_swe_model_class_can_be_disabled(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        model_class=None,
    )

    assert "model.model_class" not in agent._build_config_flags()


def test_mini_swe_openrouter_uses_native_model_class(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openrouter/minimax/minimax-m2.7",
    )

    assert agent._run_model_name == "minimax/minimax-m2.7"
    assert "-c model.model_class=openrouter" in agent._build_config_flags()
