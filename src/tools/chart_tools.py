import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx
from shapely import Point

from src.config.logger import logger


def private_public_pie(stats: Dict[str, Any], output_dir: Path) -> Path:
    """
    Build a pie / donut chart showing public vs private vs unknown cameras.
    :param stats: Summary-stats dict coming from `compute_statistics`
    :param output_dir: Directory to save the chart
    :return: A Path to the png file
    """

    total = stats["total"]
    public = stats["public_count"]
    private = stats["private_count"]
    unknown = total - public - private

    vals = [public, private, unknown]
    labels = ["Public", "Private", "Unknown"]

    fig, ax = plt.subplots(figsize=(6, 6))
    # Donut pie
    wedges, texts, autotexts = ax.pie(
        vals,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4),
    )
    ax.set(aspect="equal", title="Camera Privacy Distribution")

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "privacy_distribution.png"
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    return chart_path


def plot_zone_sensitivity(
    stats: Dict[str, Any],
    output_dir: Path,
    top_n: int = 10,
    filename: str = "zone_sensitivity.png",
) -> Path:
    """
    Build and save a stacked‐bar chart showing counts of sensitive vs non‐sensitive cameras per zone.
    :param stats: Summary-stats dict coming from `compute_statistics`
    :param output_dir: Directory to save the chart
    :param top_n: Number of zones to be plotted
    :param filename: The filename to save the chart with
    :return: Path to the png file
    """

    zone_totals: Counter = stats["zone_counts"]
    zone_sens: Counter = stats["zone_sensitivity_counts"]
    # build non‐sensitive = total − sensitive
    zone_nonsens = {z: zone_totals[z] - zone_sens.get(z, 0) for z in zone_totals}

    # pick top N zones by total cameras
    top = [z for z, _ in zone_totals.most_common(top_n)]
    other = set(zone_totals) - set(top)
    labels = top.copy()
    sens_vals = [zone_sens.get(z, 0) for z in top]
    nonsens_vals = [zone_nonsens[z] for z in top]

    if other:
        labels.append("other")
        sens_vals.append(sum(zone_sens[z] for z in other))
        nonsens_vals.append(sum(zone_nonsens[z] for z in other))

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)))
    y = list(range(len(labels)))
    ax.barh(y, nonsens_vals, label="Non-sensitive")
    ax.barh(y, sens_vals, left=nonsens_vals, label="Sensitive")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Camera count")
    ax.set_title("Sensitive vs non-sensitive cameras by zone")
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sensitivity_reasons(
    enriched_file: Union[str, Path],
    output_file: Union[str, Path],
    top_n: int = 5,
) -> Path:
    """
    Read an enriched JSON, count non-null sensitive_reason values and draw a bar chart.
    :param enriched_file: The enriched json file
    :param output_file: The Path to save the chart
    :param top_n: Number of reasons to be plotted
    :return:
    """
    enriched_path = Path(enriched_file)
    data = json.loads(enriched_path.read_text(encoding="utf-8"))

    # collect all non-null reasons
    reasons = [
        elt["analysis"]["sensitive_reason"]
        for elt in data.get("elements", [])
        if elt["analysis"].get("sensitive") and elt["analysis"].get("sensitive_reason")
    ]
    counts: Counter[str] = Counter(reasons)
    if not counts:
        raise ValueError("No sensitive_reason data found in the file")

    most_common = counts.most_common(top_n) if top_n else counts.most_common()
    labels, values = zip(*most_common)

    # plot bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values)
    ax.set_ylabel("Count")
    ax.set_title("Camera counts vs sensitivity reasons")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def _plot_top_n_bar(
    counts: "Counter",
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    top_n: int,
) -> Optional[Path]:
    """
    Render a horizontal bar chart of the top-N most common entries from
    ``counts``, bucketing the remainder as ``"other"``.

    Returns ``None`` when ``counts`` is empty — no PNG is written, so
    the dashboard sees an absent file and surfaces a captioned empty
    state instead of a degenerate ``(no data)`` bar (HF#4).

    Shared backbone for the operator and manufacturer distribution charts.
    """
    if not counts:
        logger.info(f"Skipping {out_path.name}: counter is empty")
        return None

    top = counts.most_common(top_n)
    labels = [str(label) for label, _ in top]
    values = [int(v) for _, v in top]
    other_total = sum(counts.values()) - sum(values)
    if other_total > 0:
        labels.append("other")
        values.append(int(other_total))

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)))
    y = list(range(len(labels)))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_operator_distribution(
    stats: Dict[str, Any],
    output_dir: Path,
    top_n: int = 10,
    filename: str = "operator_distribution.png",
) -> Optional[Path]:
    """
    Build and save a horizontal bar chart of camera count per operator.

    Returns ``None`` (and writes no file) when ``operator_counts`` is
    empty (HF#4) so the dashboard can render a captioned empty state
    rather than a degenerate placeholder.

    :param stats: Summary-stats dict from ``compute_statistics``.
    :param output_dir: Directory to save the chart.
    :param top_n: Number of operators to plot (rest pooled as "other").
    :param filename: Output filename. Callers pass ``operator_distribution_<city>.png``
        to namespace per-city artifacts; default kept short for
        ad-hoc invocations.
    :return: Path to the written PNG, or ``None`` when nothing to plot.
    """
    counts: Counter = stats.get("operator_counts") or Counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    return _plot_top_n_bar(
        counts,
        output_dir / filename,
        title="Cameras per operator",
        xlabel="Camera count",
        top_n=top_n,
    )


def plot_install_timeline(
    stats: Dict[str, Any],
    output_dir: Path,
    filename: str = "install_timeline.png",
) -> Optional[Path]:
    """
    Build and save a bar chart of camera count by install year.

    The ``"unknown"`` bin (cameras whose ``start_date`` was missing or
    non-parseable) is pinned to the leftmost bar so the gap in OSM
    tagging is always visible at a glance. Numeric years follow,
    sorted ascending.

    Returns ``None`` (and writes no file) when ``start_year_counts``
    is empty (HF#4).

    :param stats: Summary-stats dict from ``compute_statistics``.
        Must carry ``start_year_counts``.
    :param output_dir: Directory to save the chart.
    :param filename: Output filename. Callers pass
        ``install_timeline_<city>.png`` to namespace per-city artifacts.
    :return: Path to the written PNG, or ``None`` when nothing to plot.
    """
    counts: Counter = stats.get("start_year_counts") or Counter()

    unknown = int(counts.get("unknown", 0))
    year_items = sorted(
        ((str(k), int(v)) for k, v in counts.items() if k != "unknown"),
        key=lambda kv: kv[0],
    )

    labels: list = []
    values: list = []
    if unknown > 0:
        labels.append("unknown")
        values.append(unknown)
    labels.extend(k for k, _ in year_items)
    values.extend(v for _, v in year_items)

    if not labels:
        logger.info(f"Skipping {filename}: no install-year data")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_xlabel("Install year")
    ax.set_ylabel("Camera count")
    ax.set_title("Cameras by install year")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_manufacturer_distribution(
    stats: Dict[str, Any],
    output_dir: Path,
    top_n: int = 10,
    filename: str = "manufacturer_distribution.png",
) -> Optional[Path]:
    """
    Build and save a horizontal bar chart of camera count per manufacturer.

    Mirrors :func:`plot_operator_distribution`. ``manufacturer`` is
    sparsely tagged in OSM today, so the ``None`` (no-file) path is
    the common case for many cities (HF#4).
    """
    counts: Counter = stats.get("manufacturer_counts") or Counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    return _plot_top_n_bar(
        counts,
        output_dir / filename,
        title="Cameras per manufacturer",
        xlabel="Camera count",
        top_n=top_n,
    )


def plot_hotspots(
    hotspots_file: Union[str, Path],
    output_file: Union[str, Path],
) -> Path:
    """
    Read a GeoJSON of pre‐computed hotspots (with properties.cluster_id + count)
    and plot them against an OpenStreetMap basemap.
    :param hotspots_file: The file produced from DBSCAN
    :param output_file: The Path to save the chart
    :return:
    """
    #  load clusters
    hf = Path(hotspots_file)
    raw = json.loads(hf.read_text(encoding="utf-8"))
    feats = raw.get("features", [])

    # build a GeoDataFrame in WGS84
    rows = []
    for feat in feats:
        lon, lat = feat["geometry"]["coordinates"]
        cid = feat["properties"]["cluster_id"]
        cnt = feat["properties"]["count"]
        rows.append({"cluster_id": cid, "count": cnt, "geometry": Point(lon, lat)})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(epsg=3857)

    # prepare figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # size bubbles by count
    sizes = (gdf["count"] / gdf["count"].max()) * 1000  # normalize → marker size
    gdf.plot(
        ax=ax,
        column="cluster_id",
        cmap="tab10",
        markersize=sizes,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        legend=True,
    )

    # add Web Mercator basemap
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)

    # save
    ax.set_axis_off()
    ax.set_title("Surveillance hotspots")
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
