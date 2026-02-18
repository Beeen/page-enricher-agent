"""Observability and tracing utilities."""

import os
from contextlib import contextmanager

try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import (
        using_prompt_template,
        using_metadata,
        using_attributes,
    )
    from opentelemetry import trace

    _TRACING = True
except Exception:

    @contextmanager
    def _noop():
        yield

    def using_prompt_template(**kwargs):
        return _noop()

    def using_metadata(*args, **kwargs):
        return _noop()

    def using_attributes(*args, **kwargs):
        return _noop()

    class _FakeTrace:
        @staticmethod
        def get_current_span():
            return None

    trace = _FakeTrace()
    _TRACING = False


def init_tracing():
    """Initialize tracing at startup if configured."""
    if not _TRACING:
        return None

    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(
                space_id=space_id, api_key=api_key, project_name="signal-api"
            )
            LangChainInstrumentor().instrument(
                tracer_provider=tp,
                include_chains=True,
                include_agents=True,
                include_tools=True,
            )
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
            return tp
    except Exception:
        pass
    return None
