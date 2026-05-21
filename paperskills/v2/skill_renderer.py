"""Render PaperToSkill 2.0 evidence bundles as executable skill drafts."""

from __future__ import annotations

from datetime import datetime
from typing import List

from paperskills.v2.models import CandidatePackage, EvidenceBundle


def _bullet(items: List[str]) -> List[str]:
    return [f"- {item}" for item in items if item]


def render_skill_markdown(bundle: EvidenceBundle) -> str:
    """Render a v2 operational skill draft.

    The draft is intentionally explicit about evidence type so downstream agents
    can distinguish conceptual paper grounding from executable API grounding.
    """

    intent = bundle.task_intent
    packages = bundle.candidate_packages
    package_names = [p.package for p in packages]
    primary: CandidatePackage | None = packages[0] if packages else None

    lines: List[str] = [
        "---",
        f"name: papertoskill-v2-{intent.task_id or intent.domain or 'task'}",
        "source: papertoskill_v2",
        f"generated_at: {datetime.utcnow().isoformat()}Z",
        "evidence_model: scientific_plus_operational",
        "tags:",
        "  - papertoskill-v2",
        "  - technical-documentation",
    ]
    for package in package_names:
        lines.append(f"  - {package}")

    lines.extend(
        [
            "---",
            "",
            "## Task Intent",
            "",
            f"- Domain: {intent.domain or 'unknown'}",
            f"- Analysis intent: {intent.analysis_intent or 'unknown'}",
        ]
    )
    if intent.input_types:
        lines.extend(["- Inputs:"] + [f"  - {x}" for x in intent.input_types])
    if intent.output_types:
        lines.extend(["- Outputs:"] + [f"  - {x}" for x in intent.output_types])
    if intent.object_hints:
        lines.extend(["- Object hints:"] + [f"  - `{x}`" for x in intent.object_hints])

    lines.extend(["", "## Candidate Packages", ""])
    if packages:
        for package in packages:
            lines.append(f"- `{package.package}` ({package.ecosystem}, confidence {package.confidence:.2f}): {package.reason}")
            if package.functions:
                lines.append(f"  - Functions: {', '.join(f'`{x}`' for x in package.functions)}")
            if package.object_classes:
                lines.append(f"  - Objects: {', '.join(f'`{x}`' for x in package.object_classes)}")
    else:
        lines.append("- No package route matched; use broad scientific and technical search.")

    lines.extend(["", "## Operational Recipe", ""])
    if bundle.extracted_operations:
        for i, op in enumerate(bundle.extracted_operations, 1):
            lines.append(f"{i}. {op.step}")
            if op.function_or_object:
                lines.append(f"   - Use: `{op.function_or_object}`")
            if op.required_arguments:
                lines.append(f"   - Required arguments: {', '.join(f'`{x}`' for x in op.required_arguments)}")
            if op.input_mapping:
                lines.append(f"   - Input mapping: {op.input_mapping}")
            if op.output_mapping:
                lines.append(f"   - Output mapping: {op.output_mapping}")
            for risk in op.risks:
                lines.append(f"   - Risk: {risk}")
    elif primary:
        lines.extend(
            [
                "1. Load the package and construct the package-specific input object.",
                f"   - Primary package: `{primary.package}`",
                "2. Follow the package vignette function chain, preserving object classes.",
                "3. Coerce the final package object into the task's required output files.",
                "4. Validate output schema and rerun documentation search on any R error.",
            ]
        )

    lines.extend(["", "## Technical Evidence To Retrieve", ""])
    if bundle.technical_sources:
        for source in bundle.technical_sources:
            lines.append(f"- {source.source_type}: {source.title}")
            if source.fetch_status:
                lines.append(f"  - Fetch status: {source.fetch_status}")
            if source.source_access_level:
                lines.append(f"  - Access: {source.source_access_level}")
            if source.url:
                lines.append(f"  - URL: {source.url}")
            if source.local_path:
                lines.append(f"  - Local copy: {source.local_path}")
            if source.query:
                lines.append(f"  - Query: `{source.query}`")
            if source.functions:
                lines.append(f"  - Functions: {', '.join(f'`{x}`' for x in source.functions)}")
            if source.useful_sections:
                lines.append(f"  - Useful sections: {', '.join(source.useful_sections)}")
            if source.excerpt:
                excerpt = source.excerpt.replace("\n", "\n    ")
                lines.append(f"  - Retrieved excerpt:\n    {excerpt[:2200]}")
    else:
        lines.append("- No technical documentation sources planned.")

    lines.extend(["", "## Scientific Evidence To Retrieve", ""])
    if bundle.scientific_sources:
        for source in bundle.scientific_sources:
            lines.append(f"- {source.source_type}: {source.title}")
            if source.query:
                lines.append(f"  - Query: `{source.query}`")
    else:
        lines.append("- Search method/software/protocol papers only after package-level routes are known.")

    lines.extend(["", "## Common Failure Modes", ""])
    risks = list(intent.risk_notes)
    for package in packages:
        if package.object_classes:
            risks.append(f"`{package.package}` may require specific S4 object classes: {', '.join(package.object_classes)}.")
        if package.functions:
            risks.append(f"`{package.package}` function signatures should be checked in the reference manual before coding.")
    lines.extend(_bullet(dict.fromkeys(risks).keys() if risks else ["No risk notes available."]))

    lines.extend(["", "## Execution Feedback Search", ""])
    lines.extend(
        [
            "- If R reports `unused argument`, search the package reference manual for the exact function signature.",
            "- If R reports `no applicable method`, inspect the expected S4 object class and coercion path.",
            "- If R reports identifier/key errors, search package vignette examples for ID mapping and organism database usage.",
            "- Prefer technical documentation and Bioconductor support threads for execution errors; method papers rarely resolve API failures.",
        ]
    )

    return "\n".join(lines) + "\n"
