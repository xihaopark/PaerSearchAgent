# Architecture

PaperSearchAgent contains two related agents.

## PaperToSkill v1

V1 is paper-first. It starts with task metadata, generates method-aware paper
queries, searches PubMed, builds a candidate pool, scores papers, fetches full
text for selected sources when possible, and generates runtime `SKILL.md`
artifacts.

Core modules:

- `paperskills.library.query_generator`
- `paperskills.library.paper_search`
- `paperskills.library.paper_extraction`
- `paperskills.library.paper2skills_extractor`
- `paperskills.library.iterative_paper_retrieval`
- `paperskills.library.persistent_skill_library`

V1 is best for method/software-paper grounding. It is weaker when task success
depends on package-specific APIs, object classes, current function signatures,
or vignette workflow conventions.

## PaperToSkill v2

V2 adds an operational technical-documentation layer. It routes task text to
candidate R/Bioconductor packages, plans official package pages, vignettes,
manuals, and documentation queries, fetches technical docs when requested, and
renders a docs-grounded operational skill.

Core modules:

- `paperskills.v2.domain_router`
- `paperskills.v2.technical_docs`
- `paperskills.v2.orchestrator`
- `paperskills.v2.skill_renderer`

The intended evidence split is:

- papers answer why a method is appropriate;
- technical docs answer how to call package functions correctly.

