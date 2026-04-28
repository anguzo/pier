"""Agent backend abstraction.

A :class:`AgentBackend` describes how one agent talks to one inference
integration — for example, "Claude Code through Anthropic's API" or
"Codex through the Respan gateway." Each agent module defines its own
subclass hierarchy so agent-specific hooks (like Codex's ``config.toml``
fragments) are part of the type, not probed with ``getattr``.

Design rules:

- Purely behavioral. No declarative pass-through fields; a backend that
  only forwards a handful of env vars still overrides :meth:`build_env`
  with a single ``collect_env`` call. This keeps one place to look when
  reading what a backend does.
- Side-effect free during construction. Credential lookups happen in
  :meth:`validate` and :meth:`build_env`, which are called explicitly
  before ``run()`` and must be cheap to call repeatedly.
- Agent-specific base classes extend this one. The universal base only
  covers what every agent/backend pair needs.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar

from pier.models.agent.network import NetworkAllowlist


class MissingBackendEnvError(ValueError):
    """Raised when a backend is missing required host credentials."""


EnvGetter = Callable[[str], str | None]


@dataclass(frozen=True)
class BackendContext:
    """Inputs a backend reads when producing runtime state.

    Packaged so new inputs (e.g. Pier job metadata) can be added without
    changing every backend signature.
    """

    agent_name: str
    model_name: str | None
    runtime_model_name: str | None
    get_env: EnvGetter


def split_model_name(model_name: str) -> tuple[str, str]:
    """Split Pier's canonical ``provider/model`` id into its parts."""
    if "/" not in model_name:
        raise ValueError(
            f"Model name must be in the format provider/model, got {model_name!r}"
        )
    namespace, bare_model = model_name.split("/", 1)
    if not namespace or not bare_model:
        raise ValueError(
            f"Model name must be in the format provider/model, got {model_name!r}"
        )
    return namespace, bare_model


def strip_model_namespace(model_name: str, expected_namespace: str) -> str:
    """Return the bare model id, asserting it matches ``expected_namespace``."""
    namespace, bare_model = split_model_name(model_name)
    if namespace != expected_namespace:
        raise ValueError(
            f"Model {model_name!r} is not compatible with backend namespace "
            f"{expected_namespace!r}"
        )
    return bare_model


def collect_env(get_env: EnvGetter, names: Iterable[str]) -> dict[str, str]:
    """Copy whichever of ``names`` are set on the host into a new dict."""
    env: dict[str, str] = {}
    for name in names:
        value = get_env(name)
        if value:
            env[name] = value
    return env


def require_env(
    *,
    backend_label: str,
    get_env: EnvGetter,
    any_of: Iterable[Iterable[str]] = (),
    all_of: Iterable[str] = (),
) -> None:
    """Assert host env satisfies a backend's credential requirements.

    ``any_of`` is a list of alternative groups: each group is satisfied when
    any one of its names is set. ``all_of`` is a flat list of names that must
    all be set.

    Raises :class:`MissingBackendEnvError` with a precise, user-facing message.
    """
    missing: list[str] = []
    for alternatives in any_of:
        names = tuple(alternatives)
        if not names:
            continue
        if not any(get_env(n) for n in names):
            if len(names) == 1:
                missing.append(names[0])
            else:
                missing.append("one of " + ", ".join(names))
    for name in all_of:
        if not get_env(name):
            missing.append(name)
    if missing:
        raise MissingBackendEnvError(
            f"{backend_label} is missing required environment variables: "
            + ", ".join(missing)
        )


class AgentBackend(ABC):
    """One (agent, inference integration) pair.

    Subclasses are agent-specific. The universal surface declared here is
    what every agent needs; agent modules extend this class with additional
    hooks (for example, a Codex-specific ``config.toml`` fragment).
    """

    #: User-facing backend name selected via ``AgentConfig.backend`` or
    #: ``--backend``. Must be unique per agent. Set on the class, not instance.
    name: ClassVar[str]

    @abstractmethod
    def validate(self, ctx: BackendContext) -> None:
        """Raise if this backend cannot run for ``ctx``.

        Must be idempotent and side-effect free. Called by the agent's
        preflight before ``run()``; safe to call multiple times.
        """

    def build_env(self, ctx: BackendContext) -> dict[str, str]:
        """Return env vars to export into the agent process.

        Default: no backend-specific env. Override to pass through or rename
        host env vars for the agent's runtime.
        """
        return {}

    def runtime_model_name(self, ctx: BackendContext) -> str | None:
        """Return the exact model string to pass to the agent runtime.

        Default: honor ``runtime_model_name`` override, else return
        ``model_name`` unchanged. Override to strip the provider namespace or
        apply agent-specific transformations.
        """
        return ctx.runtime_model_name or ctx.model_name

    def network_allowlist(self) -> NetworkAllowlist | None:
        """Return the sandbox network allowlist this backend needs, if any."""
        return None


def backend_registry(*backends: AgentBackend) -> dict[str, AgentBackend]:
    """Build a backend mapping keyed by each backend's ``.name``.

    Catches duplicate names at import time and keeps dict keys from
    drifting out of sync with ``backend.name``.
    """
    registry: dict[str, AgentBackend] = {}
    for backend in backends:
        if backend.name in registry:
            raise ValueError(f"Duplicate backend registration: {backend.name!r}")
        registry[backend.name] = backend
    return registry
