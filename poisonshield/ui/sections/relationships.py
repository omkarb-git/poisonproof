"""Relationship explorer page."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .. import state
from ..components import (
    RISK_COLORS,
    callout,
    fmt,
    render_no_analysis,
    render_no_dataset,
    render_sample_detail,
)
from ..theme import hero


def render() -> None:
    hero(
        "Relationship Explorer",
        "Nearest-neighbour structure, label consistency and visual clusters.",
        kicker="PoisonShield · Relationships",
    )
    dataset = state.current_dataset()
    if dataset is None:
        render_no_dataset()
        return
    bundle = state.current_bundle()
    if bundle is None:
        render_no_analysis(dataset)
        return

    relationships = bundle.relationships
    meta = relationships.meta or {}
    st.caption(
        f"{len(relationships.sample_ids):,} feature rows · k={relationships.k} · metric={relationships.metric} · "
        f"projection={meta.get('projection_method', 'n/a')} · "
        f"clusters={meta.get('clustered_samples', 0):,} · noise={meta.get('noise_samples', 0):,}"
    )
    scores = bundle.risk.scores
    projection = relationships.projection.copy()
    projection["sample_id"] = projection.index.astype(str)
    projection["risk_category"] = scores["risk_category"].reindex(projection.index)
    projection["risk_score"] = scores["risk_score"].reindex(projection.index)
    projection["label"] = scores["label"].reindex(projection.index).astype(str)

    controls = st.columns([1.1, 1.1, 1.2, 1.4])
    with controls[0]:
        color_by = st.selectbox("Colour by", ["Risk category", "Label", "Cluster"], key="ps_rel_color")
    with controls[1]:
        risk_filter = st.multiselect(
            "Risk categories",
            ["Low", "Moderate", "High", "Critical"],
            default=["Low", "Moderate", "High", "Critical"],
            key="ps_rel_risk_filter",
        )
    with controls[2]:
        show_edges = st.checkbox("Show neighbour edges", value=True, key="ps_rel_edges")
        max_edges = st.slider("Maximum edges", 100, 5000, 1200, 100, key="ps_rel_max_edges")
    with controls[3]:
        search = st.text_input("Sample search (ID contains)", key="ps_rel_search")

    mask = projection["risk_category"].isin(risk_filter)
    if search.strip():
        mask &= projection["sample_id"].str.contains(search.strip(), case=False, regex=False)
    view = projection[mask].copy()
    if view.empty:
        st.warning("No projected samples match the current filters.")
        return

    if color_by == "Risk category":
        figure = px.scatter(
            view,
            x="x",
            y="y",
            color="risk_category",
            color_discrete_map=RISK_COLORS,
            hover_data=["sample_id", "label", "risk_score"],
        )
    elif color_by == "Label":
        figure = px.scatter(
            view,
            x="x",
            y="y",
            color="label",
            hover_data=["sample_id", "risk_category", "risk_score"],
        )
    else:
        view["cluster"] = view["cluster"].astype(str)
        figure = px.scatter(
            view,
            x="x",
            y="y",
            color="cluster",
            hover_data=["sample_id", "label", "risk_category", "risk_score"],
        )
    figure.update_traces(marker=dict(size=7, opacity=0.82, line=dict(width=0)))

    if show_edges:
        allowed = set(view.index)
        edges = relationships.edges(ids=list(allowed), max_edges=int(max_edges))
        if not edges.empty:
            xs: list[float | None] = []
            ys: list[float | None] = []
            for _, edge in edges.iterrows():
                if edge["source"] not in projection.index or edge["target"] not in projection.index:
                    continue
                xs.extend([projection.loc[edge["source"], "x"], projection.loc[edge["target"], "x"], None])
                ys.extend([projection.loc[edge["source"], "y"], projection.loc[edge["target"], "y"], None])
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="rgba(14,116,144,0.16)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        st.caption(f"{len(edges):,} undirected neighbour edges shown (capped at {max_edges:,}).")
    figure.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    st.plotly_chart(figure, width="stretch")

    st.divider()
    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.markdown("#### Neighbourhood summary")
        per_sample = relationships.per_sample
        agreement = pd.to_numeric(per_sample.get("neighbor_label_agreement"), errors="coerce")
        stats = {
            "Mean neighbour label agreement": agreement.mean() if agreement.notna().any() else None,
            "Median agreement": agreement.median() if agreement.notna().any() else None,
            "Samples with a conflict": int(per_sample.get("conflicting_neighborhood", pd.Series(dtype=bool)).fillna(False).sum()),
            "Projection variance explained": meta.get("projection_explained_variance"),
        }
        for label, value in stats.items():
            st.write(f"**{label}:** {fmt(value, 3) if isinstance(value, (int, float)) else value}")
        st.caption(
            "A neighbourhood conflict means the assigned label differs from the majority of labelled "
            "neighbours. It is supporting evidence, not confirmation of poisoning."
        )
        if meta.get("projection_note"):
            st.caption(meta["projection_note"])
        if relationships.warnings:
            for warning in relationships.warnings:
                st.warning(warning)

    with col2:
        st.markdown("#### Inspect a sample")
        options = view.sort_values("risk_score", ascending=False)["sample_id"].tolist()
        selected = st.selectbox("Sample", options, key="ps_rel_sample")
        if selected:
            render_sample_detail(bundle, selected)
            neighbours = relationships.neighbours_for(selected, limit=15)
            if not neighbours.empty:
                st.markdown("##### Neighbour labels")
                st.dataframe(
                    neighbours,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "distance": st.column_config.NumberColumn(format="%.4f"),
                        "similarity": st.column_config.NumberColumn(format="%.4f"),
                    },
                )
