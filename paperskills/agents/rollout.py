"""Vanilla rollout loop aligned with BixBench `vanilla_rollout` / ldp Trajectory."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from aviary.core import Message
from aviary.tools import ToolCall, ToolCallFunction, ToolRequestMessage
from ldp.agent import Agent
from ldp.data_structures import Trajectory, Transition
from ldp.graph import OpResult
from ldp.graph.op_utils import CallID

from .r_task_env import RTaskEvalEnv

logger = logging.getLogger(__name__)


def _sum_int_lines_safe(path: Path) -> int | None:
    """Sum one integer per non-empty line; skip lines that are not valid integers."""
    total = 0
    any_ok = False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            total += int(s)
            any_ok = True
        except ValueError:
            continue
    return total if any_ok else None


def expected_result_text_for_smoke(work_dir: Path) -> str:
    """Content for `output/result.txt` in scripted smoke: match task workspace when possible.

    Priority: first non-empty line of `evaluation/reference_sum.txt` → sum of
    `input/values.txt` → sum of `input/numbers.txt` (pilot) → legacy default ``6``.
    Malformed lines in input files are skipped. Any unexpected error falls back to ``6\\n``.
    """
    wd = Path(work_dir)
    try:
        ref = wd / "evaluation" / "reference_sum.txt"
        if ref.is_file():
            lines = [ln.strip() for ln in ref.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                return lines[0] + "\n"
        for rel in ("input/values.txt", "input/numbers.txt"):
            p = wd / rel
            if p.is_file():
                s = _sum_int_lines_safe(p)
                if s is not None:
                    return str(s) + "\n"
        return "6\n"
    except OSError as e:
        logger.warning("expected_result_text_for_smoke: %s, using fallback 6", e)
        return "6\n"


async def vanilla_r_task_rollout(
    agent: Agent[Any],
    environment: RTaskEvalEnv,
    *,
    max_steps: int = 20,
    traj_id: str | None = None,
) -> tuple[Trajectory, RTaskEvalEnv]:
    """Observe-act loop: reset env, run up to max_steps transitions."""
    obs, tools = await environment.reset()
    agent_state = await agent.init_state(tools)
    trajectory = Trajectory(traj_id=traj_id or str(uuid.uuid4()))
    # obs is Messages (list of message-like); get_asv expects list[Message]
    obs_list: list[Message] = list(obs)  # type: ignore[arg-type]

    for timestep in range(max_steps):
        action, next_agent_state, value = await agent.get_asv(agent_state, obs_list)
        if action.value is None:
            logger.warning(
                "vanilla_r_task_rollout: get_asv returned no action at timestep %s; stopping",
                timestep,
            )
            trajectory.steps = [
                *trajectory.steps,
                Transition(
                    timestep=timestep,
                    agent_state=agent_state,
                    next_agent_state=next_agent_state,
                    observation=obs_list,
                    next_observation=list(obs_list),
                    action=None,
                    reward=0.0,
                    done=False,
                    truncated=True,
                    value=value,
                ),
            ]
            break
        # vLLM sometimes returns plain Message instead of ToolRequestMessage even with tool_choice="required"
        if not isinstance(action.value, ToolRequestMessage):
            logger.error(
                "vanilla_r_task_rollout: expected ToolRequestMessage, got %s at timestep %s (model failed to emit tool calls)",
                type(action.value).__name__,
                timestep,
            )
            raise RuntimeError(
                f"Model returned {type(action.value).__name__} instead of ToolRequestMessage. "
                "This happens when the model (e.g., vLLM local Qwen) does not follow tool_choice='required' "
                "and returns plain text instead of tool calls. Consider retrying or using OpenRouter."
            )
        next_obs, reward, done, trunc = await environment.step(action.value)
        transition = Transition(
            timestep=timestep,
            agent_state=agent_state,
            next_agent_state=next_agent_state,
            observation=obs_list,
            next_observation=list(next_obs),
            action=action,
            reward=reward,
            done=done,
            truncated=trunc,
            value=value,
        )
        trajectory.steps = [*trajectory.steps, transition]
        if done or trunc:
            break
        agent_state = next_agent_state
        obs_list = list(next_obs)

    return trajectory, environment


async def save_run_artifacts(
    run_dir: Path,
    trajectory: Trajectory,
    metadata: dict[str, Any],
) -> None:
    """Write `metadata.json` and `trajectory.jsonl` (ldp `Trajectory.to_jsonl`)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    await trajectory.to_jsonl(run_dir / "trajectory.jsonl")


async def scripted_success_rollout(env: RTaskEvalEnv) -> Trajectory:
    """No-LLM smoke: write numerically correct `output/result.txt` when workspace allows, then submit_done."""
    obs, _tools = await env.reset()
    rel = "output/result.txt"
    body = expected_result_text_for_smoke(env.state.work_dir)
    msg = ToolRequestMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call_write",
                function=ToolCallFunction(
                    name="write_text_file",
                    arguments={"relative_path": rel, "content": body},
                ),
            ),
            ToolCall(
                id="call_done",
                function=ToolCallFunction(
                    name="submit_done",
                    arguments={"success": True, "summary": "scripted smoke"},
                ),
            ),
        ],
    )
    next_obs, reward, done, trunc = await env.step(msg)
    cid = CallID(
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    )
    action = OpResult(
        cid,
        "scripted_smoke",
        "scripted_success_rollout",
        msg,
    )
    traj = Trajectory(traj_id="smoke-scripted")
    traj.steps = [
        Transition(
            timestep=0,
            agent_state=None,
            next_agent_state=None,
            observation=list(obs),
            next_observation=list(next_obs),
            action=action,
            reward=reward,
            done=done,
            truncated=trunc,
            value=0.0,
        )
    ]
    return traj
