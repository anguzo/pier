from pier.agents.installed.mini_swe_agent import convert_mini_swe_agent_to_atif


def test_convert_openai_responses_usage_to_atif_metrics():
    trajectory = convert_mini_swe_agent_to_atif(
        {
            "trajectory_format": "mini-swe-agent-1.1",
            "info": {
                "mini_version": "2.2.8",
                "model_stats": {"instance_cost": 0.03, "api_calls": 1},
                "config": {"model": {"model_name": "openai/gpt-5.5"}},
            },
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {
                    "object": "response",
                    "model": "gpt-5.5-2026-04-23",
                    "usage": {
                        "input_tokens": 100,
                        "input_tokens_details": {"cached_tokens": 25},
                        "output_tokens": 40,
                        "output_tokens_details": {"reasoning_tokens": 30},
                    },
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "I will inspect the workspace.",
                                }
                            ],
                        },
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "bash",
                            "arguments": '{"command": "pwd"}',
                        },
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "/app\n",
                    "extra": {"returncode": 0, "exception_info": ""},
                },
            ],
        },
        "session",
    )

    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_completion_tokens == 40
    assert trajectory.final_metrics.total_cached_tokens == 25
    assert trajectory.final_metrics.total_cost_usd == 0.03
    assert trajectory.final_metrics.extra == {
        "total_reasoning_tokens": 30,
        "peak_context_tokens": 100,
    }

    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0].metrics is not None
    assert agent_steps[0].metrics.prompt_tokens == 100
    assert agent_steps[0].metrics.completion_tokens == 40
    assert agent_steps[0].metrics.cached_tokens == 25
    assert agent_steps[0].metrics.cost_usd == 0.03
    assert agent_steps[0].tool_calls is not None
    assert agent_steps[0].tool_calls[0].arguments == {"command": "pwd"}
    assert agent_steps[0].observation is not None
    assert "/app" in agent_steps[0].observation.results[0].content
