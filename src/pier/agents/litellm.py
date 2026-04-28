"""LiteLLM provider → env-var table.

This table is specific to agents that hand the model name directly to
LiteLLM (``mini_swe_agent`` today). It intentionally does not overlap
with OpenCode's or the per-agent Respan specs — those upstream tools
support different sets of providers and auth shapes.

The table is split into two lists:

- :data:`AUTH_ALTERNATIVES`: satisfying any one in each inner tuple
  is enough to authenticate the provider. Used by ``validate``.
- :data:`PASSTHROUGH_ENV`: names to forward to the agent if present.
  Usually a superset of ``AUTH_ALTERNATIVES`` plus base-URL/project env.

Keeping these separate means we can't accidentally read the auth
alternatives as "pass every one of these through," which was the bug
in the first pass of the refactor.
"""

from collections.abc import Iterable

from pier.agents.backends import EnvGetter, collect_env, split_model_name

#: Per-provider alternative credentials. At least one name in each inner
#: tuple must be set.

AUTH_ALTERNATIVES: dict[str, tuple[tuple[str, ...], ...]] = {
    "anthropic": (("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),),
    "openai": (("OPENAI_API_KEY",),),
    "litellm_proxy": (("OPENAI_API_KEY",),),
    "azure": (
        ("AZURE_API_KEY",),
        ("AZURE_API_BASE",),
        ("AZURE_API_VERSION",),
    ),
    "cohere": (("COHERE_API_KEY",),),
    "replicate": (("REPLICATE_API_KEY",),),
    "google": (("GEMINI_API_KEY", "GOOGLE_API_KEY"),),
    "gemini": (("GEMINI_API_KEY", "GOOGLE_API_KEY"),),
    "openrouter": (("OPENROUTER_API_KEY",),),
    "vercel_ai_gateway": (("VERCEL_AI_GATEWAY_API_KEY",),),
    "datarobot": (("DATAROBOT_API_TOKEN",),),
    "vertex_ai": (("VERTEXAI_PROJECT",),),
    "huggingface": (("HUGGINGFACE_API_KEY",),),
    "ai21": (("AI21_API_KEY",),),
    "ai21_chat": (("AI21_API_KEY",),),
    "together_ai": (("TOGETHERAI_API_KEY",),),
    "aleph_alpha": (("ALEPH_ALPHA_API_KEY",),),
    "baseten": (("BASETEN_API_KEY",),),
    "nlp_cloud": (("NLP_CLOUD_API_KEY",),),
    "bedrock": (
        ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_PROFILE"),
    ),
    "sagemaker": (("AWS_SECRET_ACCESS_KEY",),),
    "ollama": (("OLLAMA_API_BASE",),),
    "anyscale": (("ANYSCALE_API_KEY",),),
    "deepinfra": (("DEEPINFRA_API_KEY",),),
    "featherless_ai": (("FEATHERLESS_AI_API_KEY",),),
    "deepseek": (("DEEPSEEK_API_KEY",),),
    "groq": (("GROQ_API_KEY",),),
    "mistral": (("MISTRAL_API_KEY",),),
    "nvidia_nim": (("NVIDIA_NIM_API_KEY",),),
    "cerebras": (("CEREBRAS_API_KEY",),),
    "xai": (("XAI_API_KEY",),),
    "volcengine": (("VOLCENGINE_API_KEY",),),
    "codestral": (("CODESTRAL_API_KEY",),),
    "palm": (("PALM_API_KEY",),),
    "perplexity": (("PERPLEXITYAI_API_KEY",),),
    "voyage": (("VOYAGE_API_KEY",),),
    "infinity": (("INFINITY_API_KEY",),),
    "fireworks_ai": (("FIREWORKS_AI_API_KEY",),),
    "cloudflare": (("CLOUDFLARE_API_KEY",),),
    "novita": (("NOVITA_API_KEY",),),
    "nebius": (("NEBIUS_API_KEY",),),
    "dashscope": (("DASHSCOPE_API_KEY",),),
    "moonshot": (("MOONSHOT_API_KEY",),),
    "zai": (("ZAI_API_KEY",),),
}

#: Per-provider environment variables forwarded to the agent runtime when
#: set on the host. Auth alternatives are included here because the agent
#: typically needs to see the credentials; additional base-URL / project
#: env vars are listed too.
PASSTHROUGH_ENV: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"),
    "openai": ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL"),
    "litellm_proxy": ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL"),
    "azure": ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"),
    "cohere": ("COHERE_API_KEY",),
    "replicate": ("REPLICATE_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "vercel_ai_gateway": ("VERCEL_AI_GATEWAY_API_KEY",),
    "datarobot": ("DATAROBOT_API_TOKEN",),
    "vertex_ai": (
        "VERTEXAI_PROJECT",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ),
    "huggingface": ("HUGGINGFACE_API_KEY",),
    "ai21": ("AI21_API_KEY",),
    "ai21_chat": ("AI21_API_KEY",),
    "together_ai": ("TOGETHERAI_API_KEY",),
    "aleph_alpha": ("ALEPH_ALPHA_API_KEY",),
    "baseten": ("BASETEN_API_KEY",),
    "nlp_cloud": ("NLP_CLOUD_API_KEY",),
    "bedrock": (
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_REGION",
    ),
    "sagemaker": (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
    ),
    "ollama": ("OLLAMA_API_BASE",),
    "anyscale": ("ANYSCALE_API_KEY",),
    "deepinfra": ("DEEPINFRA_API_KEY",),
    "featherless_ai": ("FEATHERLESS_AI_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "nvidia_nim": ("NVIDIA_NIM_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "volcengine": ("VOLCENGINE_API_KEY",),
    "codestral": ("CODESTRAL_API_KEY",),
    "palm": ("PALM_API_KEY",),
    "perplexity": ("PERPLEXITYAI_API_KEY",),
    "voyage": ("VOYAGE_API_KEY",),
    "infinity": ("INFINITY_API_KEY",),
    "fireworks_ai": ("FIREWORKS_AI_API_KEY",),
    "cloudflare": ("CLOUDFLARE_API_KEY",),
    "novita": ("NOVITA_API_KEY",),
    "nebius": ("NEBIUS_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "zai": ("ZAI_API_KEY",),
}


def auth_alternatives_for_model(
    model_name: str,
) -> tuple[tuple[str, ...], ...]:
    """Return auth alternative groups for ``model_name``'s provider.

    Empty tuple if the provider isn't known, letting callers decide whether
    to treat that as an error.
    """
    provider, _ = split_model_name(model_name)
    return AUTH_ALTERNATIVES.get(provider, ())


def passthrough_env_for_model(
    get_env: EnvGetter, model_name: str
) -> dict[str, str]:
    """Collect forwarding env for the model's provider."""
    provider, _ = split_model_name(model_name)
    names: Iterable[str] = PASSTHROUGH_ENV.get(provider, ())
    return collect_env(get_env, names)
