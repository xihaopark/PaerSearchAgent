"""Agent layer extracted from the PaperToSkill experiments.

The imports are lazy because the ReAct execution layer depends on the original
LDP/Aviary runtime stack. Retrieval-only users can install the package without
those optional agent dependencies.
"""

__all__ = [
    "LivePaperDiscoveryEnv",
    "PaperToSkillAgent",
    "RTaskEvalEnv",
    "vanilla_r_task_rollout",
]


def __getattr__(name):
    if name == "LivePaperDiscoveryEnv":
        from paperskills.agents.live_paper_discovery_runner import LivePaperDiscoveryEnv

        return LivePaperDiscoveryEnv
    if name == "PaperToSkillAgent":
        from paperskills.agents.papertoskill_agent import PaperToSkillAgent

        return PaperToSkillAgent
    if name == "RTaskEvalEnv":
        from paperskills.agents.r_task_env import RTaskEvalEnv

        return RTaskEvalEnv
    if name == "vanilla_r_task_rollout":
        from paperskills.agents.rollout import vanilla_r_task_rollout

        return vanilla_r_task_rollout
    raise AttributeError(name)
