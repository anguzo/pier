import json
import shlex
from datetime import datetime, timezone
from typing import Any

from pier.agents.installed.base import BaseInstalledAgent, CliFlag, with_prompt_template
from pier.agents.network import allowlist_from_urls
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.name import AgentName
from pier.models.agent.network import NetworkAllowlist
from pier.models.trajectories import Agent, FinalMetrics, Metrics, Step, Trajectory
from pier.utils.trajectory_metrics import (
    extra_with_context_metrics,
    peak_context_tokens_from_steps,
    populate_context_from_final_metrics,
)
from pier.utils.trajectory_utils import format_trajectory_json


class Pi(BaseInstalledAgent):
    """The Pi agent uses the pi-coding-agent CLI to solve tasks."""

    SUPPORTS_ATIF: bool = True

    _OUTPUT_FILENAME = "pi.txt"
    _DEFAULT_PROVIDER_DOMAINS: dict[str, list[str]] = {
        "anthropic": ["api.anthropic.com"],
        "google": [".googleapis.com"],
        "groq": ["api.groq.com"],
        "mistral": ["api.mistral.ai"],
        "openai": ["api.openai.com"],
        "openrouter": ["openrouter.ai"],
        "xai": ["api.x.ai"],
    }
    CLI_FLAGS = [
        CliFlag(
            "thinking",
            cli="--thinking",
            type="enum",
            choices=["low", "medium", "high"],
        ),
    ]

    def __init__(
        self,
        *args,
        package_name: str = "@mariozechner/pi-coding-agent",
        **kwargs,
    ):
        self._package_name = package_name
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return AgentName.PI.value

    def get_version_command(self) -> str | None:
        return "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; pi --version"

    def install_spec(self) -> AgentInstallSpec:
        version_spec = f"@{self._version}" if self._version else "@latest"
        root_run = (
            "if command -v apk &> /dev/null; then"
            "  apk add --no-cache curl bash nodejs npm;"
            " elif command -v apt-get &> /dev/null; then"
            "  apt-get update && apt-get install -y curl;"
            " elif command -v yum &> /dev/null; then"
            "  yum install -y curl;"
            " else"
            '  echo "Warning: No known package manager found, assuming curl is available" >&2;'
            " fi"
        )
        agent_run = (
            "set -euo pipefail; "
            "if command -v apk &> /dev/null; then"
            f"  npm install -g --ignore-scripts {self._package_name}{version_spec};"
            " else"
            "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash &&"
            '  export NVM_DIR="$HOME/.nvm" &&'
            '  \\. "$NVM_DIR/nvm.sh" || true &&'
            "  command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } &&"
            "  nvm install 22 && nvm alias default 22 && npm -v &&"
            f"  npm install -g --ignore-scripts {self._package_name}{version_spec};"
            " fi && "
            "pi --version"
        )
        symlink_run = (
            "for bin in node pi; do"
            '  BIN_PATH="$(which "$bin" 2>/dev/null || true)";'
            '  if [ -n "$BIN_PATH" ] && [ "$BIN_PATH" != "/usr/local/bin/$bin" ]; then'
            '    ln -sf "$BIN_PATH" "/usr/local/bin/$bin";'
            "  fi;"
            " done"
        )
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=root_run,
                ),
                InstallStep(user="agent", run=agent_run),
                InstallStep(user="root", run=symlink_run),
            ],
            verification_command=self.get_version_command(),
        )

    def network_allowlist(self) -> NetworkAllowlist:
        provider = self._provider_name()
        urls = []
        for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
            if value := self._get_env(key):
                urls.append(value)
        return allowlist_from_urls(
            urls,
            default_domains=self._DEFAULT_PROVIDER_DOMAINS.get(provider, []),
        )

    def _provider_name(self) -> str:
        if self.model_name and "/" in self.model_name:
            return self.model_name.split("/", 1)[0]
        return "openai"

    def _model_args(self) -> str:
        if not self.model_name:
            raise ValueError("model_name is required for pi")
        if "/" in self.model_name:
            return f"--model {shlex.quote(self.model_name)}"
        return f"--provider openai --model {shlex.quote(self.model_name)}"

    def _parse_stdout(self) -> list[dict[str, Any]]:
        output_path = self.logs_dir / self._OUTPUT_FILENAME
        if not output_path.exists():
            return []

        events: list[dict[str, Any]] = []
        for line in output_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    @staticmethod
    def _millis_to_iso(timestamp_ms: int | float | None) -> str | None:
        if timestamp_ms is None:
            return None
        try:
            return datetime.fromtimestamp(
                timestamp_ms / 1000, tz=timezone.utc
            ).isoformat()
        except (OSError, ValueError, OverflowError):
            return None

    @staticmethod
    def _extract_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [Pi._extract_text(item) for item in value]
            return "".join(part for part in parts if part)
        if isinstance(value, dict):
            for key in ("text", "content", "message", "delta"):
                text = Pi._extract_text(value.get(key))
                if text:
                    return text
        return ""

    @staticmethod
    def _as_int(value: Any) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _as_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _first_value(data: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return None

    @staticmethod
    def _usage_from_event(event: dict[str, Any]) -> dict[str, Any]:
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        usage = (
            event.get("usage")
            or event.get("tokens")
            or event.get("metrics")
            or message.get("usage")
            or message.get("tokens")
            or message.get("metrics")
            or {}
        )
        return usage if isinstance(usage, dict) else {}

    @staticmethod
    def _cost_from_event(event: dict[str, Any], usage: dict[str, Any]) -> float:
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        return Pi._as_float(
            Pi._first_value(
                usage,
                "cost_usd",
                "costUsd",
                "costUSD",
                "cost",
                "totalCost",
                "total_cost",
            )
            or Pi._first_value(
                message,
                "cost_usd",
                "costUsd",
                "costUSD",
                "cost",
                "totalCost",
                "total_cost",
            )
            or Pi._first_value(
                event,
                "cost_usd",
                "costUsd",
                "costUSD",
                "cost",
                "totalCost",
                "total_cost",
            )
        )

    @staticmethod
    def _candidate_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        message_end = [event for event in events if event.get("type") == "message_end"]
        if message_end:
            return message_end
        turn_end = [event for event in events if event.get("type") == "turn_end"]
        if turn_end:
            return turn_end
        agent_end_messages: list[dict[str, Any]] = []
        for event in events:
            if event.get("type") != "agent_end":
                continue
            for message in event.get("messages") or []:
                if isinstance(message, dict):
                    agent_end_messages.append({"type": "message_end", "message": message})
        return agent_end_messages or events

    def _convert_events_to_trajectory(
        self, events: list[dict[str, Any]]
    ) -> Trajectory | None:
        steps: list[Step] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        total_cost = 0.0
        session_id = "unknown"

        for event in events:
            if isinstance(event.get("session_id"), str):
                session_id = event["session_id"]
            elif isinstance(event.get("sessionId"), str):
                session_id = event["sessionId"]
            elif event.get("type") == "session" and isinstance(event.get("id"), str):
                session_id = event["id"]

        for event in self._candidate_events(events):
            message_obj = (
                event.get("message") if isinstance(event.get("message"), dict) else {}
            )

            event_type = str(event.get("type") or event.get("event") or "").lower()
            role = str(event.get("role") or message_obj.get("role") or "").lower()
            if role not in {"assistant", "agent"} and not any(
                marker in event_type for marker in ("assistant", "agent", "message")
            ):
                continue

            message = self._extract_text(message_obj or event)
            usage = self._usage_from_event(event)
            prompt_tokens = int(
                self._as_int(
                    self._first_value(
                        usage,
                        "prompt_tokens",
                        "promptTokens",
                        "input_tokens",
                        "inputTokens",
                        "input",
                    )
                )
            )
            completion_tokens = int(
                self._as_int(
                    self._first_value(
                        usage,
                        "completion_tokens",
                        "completionTokens",
                        "output_tokens",
                        "outputTokens",
                        "output",
                    )
                )
            )
            cache_read_tokens = self._as_int(
                self._first_value(
                    usage,
                    "cached_tokens",
                    "cachedTokens",
                    "cache_read_tokens",
                    "cacheReadTokens",
                    "cacheRead",
                )
            )
            cache_write_tokens = self._as_int(
                self._first_value(
                    usage,
                    "cache_write_tokens",
                    "cacheWriteTokens",
                    "cacheWrite",
                )
            )
            cost = self._cost_from_event(event, usage)

            if not message and not prompt_tokens and not completion_tokens and not cost:
                continue

            total_prompt_tokens += prompt_tokens + cache_read_tokens
            total_completion_tokens += completion_tokens
            total_cached_tokens += cache_read_tokens
            total_cost += cost

            timestamp = self._millis_to_iso(
                event.get("timestamp_ms") or event.get("timestampMs")
            )
            metrics = None
            if prompt_tokens or completion_tokens or cost:
                extra = {
                    k: v
                    for k, v in {
                        "cache_write_tokens": cache_write_tokens,
                    }.items()
                    if v
                }
                metrics = Metrics(
                    prompt_tokens=(prompt_tokens + cache_read_tokens) or None,
                    completion_tokens=completion_tokens or None,
                    cached_tokens=cache_read_tokens or None,
                    cost_usd=cost or None,
                    extra=extra or None,
                )

            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=timestamp,
                    source="agent",
                    message=message,
                    model_name=self.model_name,
                    llm_call_count=1,
                    metrics=metrics,
                )
            )

        if not steps:
            return None

        final_metrics = FinalMetrics(
            total_prompt_tokens=total_prompt_tokens or None,
            total_completion_tokens=total_completion_tokens or None,
            total_cached_tokens=total_cached_tokens or None,
            total_cost_usd=total_cost or None,
            total_steps=len(steps),
            extra=extra_with_context_metrics(
                None,
                peak_context_tokens=peak_context_tokens_from_steps(steps),
                summarization_count=None,
            ),
        )
        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=final_metrics,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        events = self._parse_stdout()
        if not events:
            return

        try:
            trajectory = self._convert_events_to_trajectory(events)
        except Exception:
            self.logger.exception("Failed to convert pi events to trajectory")
            return

        if not trajectory:
            return

        trajectory_path = self.logs_dir / "trajectory.json"
        try:
            trajectory_path.write_text(
                format_trajectory_json(trajectory.to_json_dict())
            )
        except OSError as exc:
            self.logger.debug(f"Failed to write trajectory file {trajectory_path}: {exc}")

        if trajectory.final_metrics:
            populate_context_from_final_metrics(context, trajectory.final_metrics)

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError("model_name is required for pi")

        provider = self._provider_name()
        env = self.build_process_env(
            {
                "PI_SKIP_VERSION_CHECK": "1",
                "PI_TELEMETRY": "0",
            }
        )
        if provider == "openai":
            for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
                if value := self._get_env(key):
                    env[key] = value

        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""

        await self.exec_as_agent(
            environment,
            command=(
                "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                f"pi --mode json {self._model_args()} "
                f"{cli_flags_arg}--no-session -p {shlex.quote(instruction)} "
                f"2>&1 </dev/null | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
            ),
            env=env,
        )
