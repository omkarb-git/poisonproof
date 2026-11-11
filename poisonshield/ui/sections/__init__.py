"""Navigation and section dispatch for the PoisonShield UI."""

from __future__ import annotations

from . import (
    evaluation,
    image_analysis,
    inspection,
    label_review,
    overview,
    relationships,
    reports,
    sanitisation,
    suspicious,
    tabular_analysis,
    upload,
)

SECTIONS = [
    "Overview",
    "Upload",
    "Inspection",
    "Tabular Analysis",
    "Image Analysis",
    "Relationship Explorer",
    "Suspicious Samples",
    "Label Review",
    "Sanitisation",
    "Model / Quality Evaluation",
    "Reports / Export",
]

RENDERERS = {
    "Overview": overview.render,
    "Upload": upload.render_upload,
    "Inspection": inspection.render,
    "Tabular Analysis": tabular_analysis.render,
    "Image Analysis": image_analysis.render,
    "Relationship Explorer": relationships.render,
    "Suspicious Samples": suspicious.render,
    "Label Review": label_review.render,
    "Sanitisation": sanitisation.render,
    "Model / Quality Evaluation": evaluation.render,
    "Reports / Export": reports.render,
}


def render_section(section: str) -> None:
    renderer = RENDERERS.get(section)
    if renderer is None:
        renderer = overview.render
    renderer()
