"""Regenerate every figure and table from results/raw/.

    sbatch slurm/run_analysis.sh          # the one documented command

Reads raw artifacts only. Writes results/figures/ and results/RESULTS.md.
Nothing here transcribes a number: if a value is not in an artifact, the figure
shows a gap rather than an estimate.

Plot conventions (workspace spec): 300 dpi, axis units labelled, hardware
and replica topology in the caption, N and spread shown, median with min/max
rather than a bare mean.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import (  # noqa: E402
    Phase,
    group,
    hardware_caption,
    load_failure_trials,
    load_measurements,
    prefix_group_counts,
)

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "results" / "figures"

POLICY_ORDER = [
    "round_robin",
    "least_loaded",
    "prefix_affinity",
    "prefix_affinity_bounded",
]
POLICY_LABEL = {
    "round_robin": "round robin",
    "least_loaded": "least loaded",
    "prefix_affinity": "prefix affinity",
    "prefix_affinity_bounded": "prefix affinity (bounded)",
}
# Colourblind-safe, distinguishable in greyscale by marker as well as hue.
POLICY_COLOR = {
    "round_robin": "#4C72B0",
    "least_loaded": "#DD8452",
    "prefix_affinity": "#C44E52",
    "prefix_affinity_bounded": "#55A868",
}
POLICY_MARKER = {
    "round_robin": "o",
    "least_loaded": "s",
    "prefix_affinity": "^",
    "prefix_affinity_bounded": "D",
}


def med_lo_hi(values: Sequence[float | None]) -> tuple[float, float, float] | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return float(np.median(a)), float(a.min()), float(a.max())


def _series(
    phases: list[Phase], policy: str, ratio: float, metric
) -> tuple[list[float], list[float], list[float], list[float], list[int]]:
    """(rates, medians, lo, hi, n) for one policy at one sharing ratio."""
    g = group([p for p in phases if p.policy == policy and p.ratio == ratio])
    rates = sorted({k[2] for k in g})
    xs, med, lo, hi, ns = [], [], [], [], []
    for r in rates:
        reps = g[(policy, ratio, r)]
        s = med_lo_hi([metric(p) for p in reps])
        if s is None:
            continue
        xs.append(r)
        med.append(s[0])
        lo.append(s[1])
        hi.append(s[2])
        ns.append(len(reps))
    return xs, med, lo, hi, ns


# Appended to every figure name while a non-default workload is being drawn,
# so the single-prefix and many-prefix experiments never overwrite each other.
SUFFIX = ""


def _finish(fig: plt.Figure, path: Path, caption: str) -> None:
    path = path.with_name(f"{path.stem}{SUFFIX}{path.suffix}")
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=7, wrap=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


def fig_ttft_vs_rate(phases: list[Phase], caption: str) -> None:
    """TTFT p50 and p95 against offered rate, one panel per sharing ratio.

    The headline figure. Prefix affinity is claimed to help TTFT at high sharing
    and to degrade under load imbalance, so the sharing ratio has to be a panel
    axis rather than pooled: pooling would average the regime where it helps
    with the regime where it hurts and report that nothing happens.
    """
    ratios = sorted({p.ratio for p in phases})
    if not ratios:
        return
    variants = (
        ("p50", "median", "", lambda p, s: p.ttft(s), "from scheduled arrival"),
        ("p95", "p95", "", lambda p, s: p.ttft(s), "from scheduled arrival"),
        ("p50", "median", "_dispatch", lambda p, s: p.ttft_from_dispatch(s), "from actual dispatch"),
        ("p95", "p95", "_dispatch", lambda p, s: p.ttft_from_dispatch(s), "from actual dispatch"),
    )
    for stat, label, suffix, getter, clock in variants:
        fig, axes = plt.subplots(
            1, len(ratios), figsize=(4.0 * len(ratios), 3.8), sharey=True, squeeze=False
        )
        for ax, ratio in zip(axes[0], ratios, strict=True):
            for pol in POLICY_ORDER:
                xs, med, lo, hi, ns = _series(
                    phases, pol, ratio, lambda p, s=stat, g=getter: g(p, s)
                )
                if not xs:
                    continue
                ax.plot(
                    xs, med, marker=POLICY_MARKER[pol], color=POLICY_COLOR[pol],
                    label=f"{POLICY_LABEL[pol]} (N={min(ns)}-{max(ns)})", lw=1.6, ms=5,
                )
                ax.fill_between(xs, lo, hi, color=POLICY_COLOR[pol], alpha=0.16, lw=0)
            ax.set_title(f"shared-prefix ratio {ratio:g}", fontsize=10)
            ax.set_xlabel("offered request rate (requests/s)")
            ax.grid(alpha=0.3, ls=":")
        axes[0][0].set_ylabel(f"TTFT {label} (s), {clock}")
        axes[0][0].set_yscale("log")
        axes[0][-1].legend(fontsize=7, framealpha=0.9)
        fig.suptitle(
            f"Time to first token ({label}, {clock}) vs offered load, by routing policy",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _finish(
            fig,
            FIGS / f"ttft_{stat}{suffix}_vs_rate.png",
            "Bands show min-max over repetitions. Scheduled-arrival TTFT includes client "
            "dispatch lag (open-loop convention); dispatch TTFT removes it. "
            f"{caption}",
        )


def fig_fairness(phases: list[Phase], caption: str) -> None:
    """Per-replica load fairness. The load-imbalance failure mode, quantified.

    Two metrics side by side on purpose. Jain's index compresses near 1.0 for
    mild imbalance and is easy to misread as "fair"; the peak-to-mean ratio
    keeps resolving and says directly how overloaded the hottest replica is.
    """
    ratios = sorted({p.ratio for p in phases})
    if not ratios:
        return
    for metric, name, ylab, ref in (
        (lambda p: p.jains_index, "jains_index", "Jain's fairness index", 1.0),
        (lambda p: p.max_over_mean, "max_over_mean", "peak / mean replica load", 1.0),
    ):
        fig, axes = plt.subplots(
            1, len(ratios), figsize=(4.0 * len(ratios), 3.6), sharey=True, squeeze=False
        )
        for ax, ratio in zip(axes[0], ratios, strict=True):
            for pol in POLICY_ORDER:
                xs, med, lo, hi, ns = _series(phases, pol, ratio, metric)
                if not xs:
                    continue
                ax.plot(
                    xs, med, marker=POLICY_MARKER[pol], color=POLICY_COLOR[pol],
                    label=POLICY_LABEL[pol], lw=1.6, ms=5,
                )
                ax.fill_between(xs, lo, hi, color=POLICY_COLOR[pol], alpha=0.16, lw=0)
            ax.axhline(ref, color="k", ls="--", lw=0.8, alpha=0.5)
            ax.set_title(f"shared-prefix ratio {ratio:g}", fontsize=10)
            ax.set_xlabel("offered request rate (requests/s)")
            ax.grid(alpha=0.3, ls=":")
        axes[0][0].set_ylabel(ylab)
        axes[0][-1].legend(fontsize=7)
        fig.suptitle(f"Per-replica load distribution: {ylab}", fontsize=12)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _finish(
            fig,
            FIGS / f"fairness_{name}.png",
            f"Dashed line is a perfectly even split. Bands are min-max over repetitions. {caption}",
        )


def fig_prefix_cache(phases: list[Phase], caption: str) -> None:
    """Fleet-wide prefix cache hit rate vs sharing ratio, per policy.

    This is the mechanism figure: if prefix affinity works, its hit rate must
    exceed round robin's at the same sharing ratio, because that is the only
    thing it does differently. A TTFT win without a hit-rate win would mean the
    latency difference came from somewhere else.

    The theoretical bound from P1's generator is drawn on the same axes; without
    it the measured gap is just a number with nothing to attribute it against.
    """
    rates = sorted({p.rate for p in phases})
    if not rates:
        return
    fig, axes = plt.subplots(
        1, len(rates), figsize=(3.8 * len(rates), 3.6), sharey=True, squeeze=False
    )
    for ax, rate in zip(axes[0], rates, strict=True):
        for pol in POLICY_ORDER:
            sel = [p for p in phases if p.policy == pol and p.rate == rate]
            g = group(sel)
            ratios = sorted({k[1] for k in g})
            xs, med, lo, hi = [], [], [], []
            for r in ratios:
                s = med_lo_hi([p.fleet_prefix_hit_rate for p in g[(pol, r, rate)]])
                if s is None:
                    continue
                xs.append(r)
                med.append(s[0])
                lo.append(s[1])
                hi.append(s[2])
            if not xs:
                continue
            ax.plot(
                xs, med, marker=POLICY_MARKER[pol], color=POLICY_COLOR[pol],
                label=POLICY_LABEL[pol], lw=1.6, ms=5,
            )
            ax.fill_between(xs, lo, hi, color=POLICY_COLOR[pol], alpha=0.16, lw=0)
        # Theoretical bound, taken from the artifacts rather than recomputed.
        bsel = [p for p in phases if p.rate == rate and p.theoretical_bound is not None]
        if bsel:
            bg: dict[float, list[float]] = {}
            for p in bsel:
                bg.setdefault(p.ratio, []).append(p.theoretical_bound)
            bx = sorted(bg)
            ax.plot(
                bx, [float(np.median(bg[r])) for r in bx],
                color="k", ls="--", lw=1.0, alpha=0.6, label="theoretical bound",
            )
        ax.set_title(f"{rate:g} req/s", fontsize=10)
        ax.set_xlabel("shared-prefix ratio")
        ax.grid(alpha=0.3, ls=":")
    axes[0][0].set_ylabel("fleet prefix cache hit rate")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Fleet-wide prefix cache hit rate: cache locality is a routing property", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _finish(
        fig,
        FIGS / "prefix_cache_hit_rate.png",
        "Hit rate is total hits / total queries summed across replicas (not a mean of "
        f"per-replica rates, which would misweight an intentionally uneven split). {caption}",
    )


def fig_throughput_error(phases: list[Phase], caption: str) -> None:
    """Achieved throughput and error rate vs offered rate.

    Reported together because they are the same fact seen two ways: past
    saturation, achieved throughput flattens and the excess shows up as 429s.
    A throughput curve alone would look like a clean plateau with no cost.
    """
    ratios = sorted({p.ratio for p in phases})
    if not ratios:
        return
    fig, axes = plt.subplots(2, len(ratios), figsize=(4.0 * len(ratios), 6.4), sharex="col", squeeze=False)
    for col, ratio in enumerate(ratios):
        for pol in POLICY_ORDER:
            xs, med, lo, hi, _ = _series(phases, pol, ratio, lambda p: p.throughput_req_s)
            if xs:
                axes[0][col].plot(xs, med, marker=POLICY_MARKER[pol], color=POLICY_COLOR[pol],
                                  label=POLICY_LABEL[pol], lw=1.6, ms=5)
                axes[0][col].fill_between(xs, lo, hi, color=POLICY_COLOR[pol], alpha=0.16, lw=0)
            xs, med, lo, hi, _ = _series(phases, pol, ratio, lambda p: p.error_rate)
            if xs:
                axes[1][col].plot(xs, med, marker=POLICY_MARKER[pol], color=POLICY_COLOR[pol], lw=1.6, ms=5)
                axes[1][col].fill_between(xs, lo, hi, color=POLICY_COLOR[pol], alpha=0.16, lw=0)
        axes[0][col].set_title(f"shared-prefix ratio {ratio:g}", fontsize=10)
        axes[1][col].set_xlabel("offered request rate (requests/s)")
        for row in (0, 1):
            axes[row][col].grid(alpha=0.3, ls=":")
        # y=x reference: below saturation, achieved should track offered.
        if xs:
            lim = [min(xs), max(xs)]
            axes[0][col].plot(lim, lim, color="k", ls=":", lw=0.8, alpha=0.5)
    axes[0][0].set_ylabel("achieved throughput (requests/s)")
    axes[1][0].set_ylabel("error rate (fraction)")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Achieved throughput and error rate vs offered load", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _finish(
        fig,
        FIGS / "throughput_and_errors.png",
        "Dotted diagonal is achieved = offered. Departure from it marks saturation; "
        f"the error-rate panel shows what the excess became. {caption}",
    )


def fig_spill(phases: list[Phase], caption: str) -> None:
    """How often the bounded policy's load guard actually fired.

    Without this, a difference between the two affinity policies is an
    observation; with it, it is an explanation.
    """
    sel = [p for p in phases if p.policy == "prefix_affinity_bounded"]
    if not sel:
        return
    ratios = sorted({p.ratio for p in sel})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cmap = plt.get_cmap("viridis")
    for i, ratio in enumerate(ratios):
        g = group([p for p in sel if p.ratio == ratio])
        rates = sorted({k[2] for k in g})
        xs, med, lo, hi = [], [], [], []
        for r in rates:
            s = med_lo_hi([p.spill_fraction for p in g[("prefix_affinity_bounded", ratio, r)]])
            if s is None:
                continue
            xs.append(r)
            med.append(s[0])
            lo.append(s[1])
            hi.append(s[2])
        if not xs:
            continue
        c = cmap(i / max(len(ratios) - 1, 1))
        ax.plot(xs, med, marker="o", color=c, label=f"ratio {ratio:g}", lw=1.6, ms=5)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.16, lw=0)
    ax.set_xlabel("offered request rate (requests/s)")
    ax.set_ylabel("fraction of decisions that spilled off the affinity primary")
    ax.set_title("prefix_affinity_bounded: how often the load guard fired", fontsize=12)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _finish(
        fig,
        FIGS / "bounded_spill_fraction.png",
        "A spill fraction near 0 means the bounded policy behaved like plain affinity; "
        f"near 1 means it behaved like least-loaded. {caption}",
    )


def fig_streaming_evidence(phases: list[Phase], caption: str) -> None:
    """Client-side proof that responses streamed under real load.

    The discriminator is TTFT/e2e. A buffering proxy delivers nothing until
    generation finishes, so the ratio would sit at ~1.0 for every request
    regardless of policy or load. This is the under-load half of the
    non-buffering verification; tests/streaming.rs is the deterministic half.
    """
    vals = [(p.rate, p.streaming_ttft_over_e2e, p.chunks_per_request) for p in phases]
    vals = [(r, t, c) for r, t, c in vals if t is not None]
    if not vals:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    rates = sorted({r for r, _, _ in vals})
    axes[0].boxplot(
        [[t for r, t, _ in vals if r == rr] for rr in rates],
        tick_labels=[f"{rr:g}" for rr in rates],
    )
    axes[0].axhline(1.0, color="r", ls="--", lw=1.0)
    axes[0].text(
        0.02, 0.92, "1.0 = fully buffered", transform=axes[0].transAxes, color="r", fontsize=8
    )
    axes[0].set_xlabel("offered request rate (requests/s)")
    axes[0].set_ylabel("median TTFT / end-to-end latency")
    axes[0].set_title("Streaming: first token arrives early", fontsize=10)
    axes[0].grid(alpha=0.3, ls=":")

    cvals = [(r, c) for r, _, c in vals if c is not None]
    if cvals:
        axes[1].boxplot(
            [[c for r, c in cvals if r == rr] for rr in rates],
            tick_labels=[f"{rr:g}" for rr in rates],
        )
        axes[1].set_xlabel("offered request rate (requests/s)")
        axes[1].set_ylabel("median SSE chunks per request")
        axes[1].set_title("Streaming: response framing preserved", fontsize=10)
        axes[1].grid(alpha=0.3, ls=":")
    fig.suptitle("SSE passthrough is non-buffering under load", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _finish(
        fig,
        FIGS / "streaming_non_buffering.png",
        "A buffering proxy would pin the left panel at 1.0 and the right panel near 1 chunk. "
        f"{caption}",
    )


def _failure_row(t: dict[str, Any]) -> dict[str, Any]:
    """Per-trial quantities, every one read or counted from the artifact."""
    killed = t["full_config_dict"]["killed_replica"]
    s = t.get("summary", {})
    rec = t.get("recovery", {})
    b = s.get("buckets_relative_to_kill", {})
    after = [r for r in t.get("per_request", []) if rec.get("recovery_offset_from_phase_start_s") is not None
             and r["scheduled_at"] >= rec["recovery_offset_from_phase_start_s"]]
    share_back = (sum(1 for r in after if r.get("replica") == killed) / len(after)) if after else None
    return {
        "policy": t["full_config_dict"]["phase"]["policy"],
        "detection_s": t.get("detection", {}).get("detection_latency_s"),
        "readmission_s": rec.get("restart_to_readmission_s"),
        "ejected_s": rec.get("ejected_duration_s"),
        # Scheduled before the kill but failed: in flight on the killed replica.
        "failed_before_kill": b.get("before_kill", {}).get("failed"),
        "failed_during_detection": b.get("during_detection_window", {}).get("failed"),
        "failed_after_ejection": b.get("after_ejection", {}).get("failed"),
        "failed_after_recovery": b.get("after_recovery", {}).get("failed"),
        "error_rate": s.get("error_rate"),
        "killed_share_after_recovery": share_back,
        "restart_rc": t.get("restart_event", {}).get("restart_returncode"),
    }


def fig_failure(trials: list[dict[str, Any]], caption: str) -> None:
    """Failed requests over time around the kill, one panel per policy."""
    if not trials:
        return
    pols = [p for p in POLICY_ORDER if any(t["full_config_dict"]["phase"]["policy"] == p for t in trials)]
    fig, axes = plt.subplots(len(pols), 1, figsize=(8.0, 2.2 * len(pols)), sharex=True, squeeze=False)
    binw = 2.0
    for ax, pol in zip(axes[:, 0], pols, strict=True):
        for t in [t for t in trials if t["full_config_dict"]["phase"]["policy"] == pol]:
            k = t["summary"].get("kill_offset_from_phase_start_s")
            if k is None:
                continue
            recs = t.get("per_request", [])
            if not recs:
                continue
            xs = np.array([r["scheduled_at"] - k for r in recs])
            fails = np.array([0 if r.get("ok") else 1 for r in recs])
            edges = np.arange(np.floor(xs.min()), np.ceil(xs.max()) + binw, binw)
            tot, _ = np.histogram(xs, edges)
            bad, _ = np.histogram(xs, edges, weights=fails)
            frac = np.divide(bad, tot, out=np.full_like(bad, np.nan, dtype=float), where=tot > 0)
            rep = t.get("repetition_index")
            line, = ax.plot(edges[:-1] + binw / 2, frac, lw=1.0, alpha=0.8, label=f"rep {rep}")
            det = t.get("detection", {}).get("detection_latency_s")
            if det is not None:
                ax.axvline(det, color=line.get_color(), ls=":", lw=0.9)
            ro = t.get("recovery", {}).get("recovery_offset_from_phase_start_s")
            if ro is not None:
                ax.axvline(ro - k, color=line.get_color(), ls="--", lw=0.9)
        ax.axvline(0, color="k", lw=1.0)
        ax.set_ylabel("failed fraction")
        ax.set_title(POLICY_LABEL[pol], fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3, ls=":")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1][0].set_xlabel("request scheduled time relative to SIGKILL (s)")
    fig.suptitle("Replica failure: failed requests around the kill, ejection and readmission", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _finish(
        fig,
        FIGS / "failure_mode_timeline.png",
        f"{binw:g} s bins by scheduled arrival. Solid black: SIGKILL. Dotted: router ejected the "
        f"replica. Dashed: router readmitted the relaunched replica. {caption}",
    )


def failure_md(trials: list[dict[str, Any]], finv: dict[str, Any]) -> list[str]:
    out = ["## Replica failure and recovery", ""]
    if not trials:
        return out + ["**Not yet measured.** No failure-mode artifacts present.", ""]
    t0 = trials[0]
    fc = t0["full_config_dict"]
    out += [
        f"Slurm job {finv['slurm_job_id']}, {finv['trials']} trials. Replica "
        f"`{fc['killed_replica']}` SIGKILLed (whole process tree) {fc['kill_after_s']:g} s into "
        f"an open-loop phase at {fc['phase']['request_rate']:g} req/s, shared-prefix ratio "
        f"{fc['phase']['shared_prefix_ratio']:g}, then relaunched "
        f"{t0.get('recovery', {}).get('restart_after_kill_s')} s after the kill completed. "
        f"Health check every {t0['detection']['health_interval_ms']} ms, ejection after "
        f"{t0['detection']['unhealthy_threshold']} consecutive failures. The first failing check "
        f"can land just after the kill, so detection is bounded below by about "
        f"{(t0['detection']['unhealthy_threshold'] - 1) * t0['detection']['health_interval_ms'] / 1000:g} s "
        f"(the artifacts' `expected_lower_bound_s` of "
        f"{t0['detection']['expected_lower_bound_s']:g} s overstates it by one interval). "
        "Each trial starts a fresh fleet.",
        "",
        "`readmission` is from the relaunch command to the router readmitting the replica "
        "(dominated by vLLM cold start). `share back` is the killed replica's share of requests "
        "scheduled after readmission (0.25 would be an even split).",
        "",
        "| policy | N | detection (s) | readmission (s) | ejected for (s) | failed, scheduled before kill "
        "| failed, kill to ejection | failed while ejected | failed after readmission | error rate | share back |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = [_failure_row(t) for t in trials]

    def f(vals: list[Any], nd: int) -> str:
        s = med_lo_hi(vals)
        return "not measured" if s is None else f"{s[0]:.{nd}f} ({s[1]:.{nd}f}-{s[2]:.{nd}f})"

    for pol in POLICY_ORDER:
        rs = [r for r in rows if r["policy"] == pol]
        if not rs:
            continue
        out.append(
            f"| {POLICY_LABEL[pol]} | {len(rs)} | {f([r['detection_s'] for r in rs], 2)} "
            f"| {f([r['readmission_s'] for r in rs], 1)} | {f([r['ejected_s'] for r in rs], 1)} "
            f"| {f([r['failed_before_kill'] for r in rs], 0)} "
            f"| {f([r['failed_during_detection'] for r in rs], 0)} "
            f"| {f([r['failed_after_ejection'] for r in rs], 0)} "
            f"| {f([r['failed_after_recovery'] for r in rs], 0)} "
            f"| {f([r['error_rate'] for r in rs], 4)} "
            f"| {f([r['killed_share_after_recovery'] for r in rs], 3)} |"
        )
    bad = [r for r in rows if r["restart_rc"] not in (0, None)]
    if bad:
        out += ["", f"{len(bad)} trial(s) had a non-zero relaunch return code; see the artifacts."]
    out.append("")
    return out


def results_section(
    phases: list[Phase],
    inventory: dict[str, Any],
    caption: str,
) -> list[str]:
    """One experiment's tables. Headings are demoted by the caller."""
    out = [
        f"**Hardware / topology:** {caption}",
        "",
        "## Inventory",
        "",
        "| quantity | value |",
        "|---|---|",
    ]
    for k in (
        "run_id", "slurm_job_id",
        "artifacts_total", "usable", "warmup_excluded",
        "server_problem_excluded", "client_lag_flagged_kept",
        "min_repetitions", "max_repetitions",
    ):
        if k in inventory:
            out.append(f"| {k.replace('_', ' ')} | {inventory[k]} |")
    out += [
        f"| policies | {', '.join(inventory['policies'])} |",
        f"| shared-prefix ratios | {', '.join(str(r) for r in inventory['ratios'])} |",
        f"| offered rates (req/s) | {', '.join(str(r) for r in inventory['rates'])} |",
        "",
    ]

    if inventory.get("superseded_runs"):
        out += [
            "### Superseded runs (in results/raw/, not used here)",
            "",
            "| run_id | Slurm job | artifacts |",
            "|---|---|---|",
        ]
        for rid, d in inventory["superseded_runs"].items():
            out.append(f"| `{rid}` | {d['slurm_job_id']} | {d['artifacts']} |")
        out.append("")

    if inventory.get("degraded_detail"):
        out += [
            "### Excluded configurations",
            "",
            "Recorded in `results/raw/` and excluded from the figures. A failed or "
            "degraded configuration is data, not a gap.",
            "",
            "| phase | problems |",
            "|---|---|",
        ]
        for d in inventory["degraded_detail"][:60]:
            out.append(f"| `{d['phase']}` | {d['problems']} |")
        out.append("")

    if not phases:
        out += ["## Measurements", "", "**Not yet measured.** No usable artifacts present.", ""]
        return out

    g = group(phases)
    out += [
        "## Per-configuration summary",
        "",
        "Median over repetitions, with (min-max) where spread is shown. "
        "`hit rate` is fleet-wide prefix cache hits/queries. "
        "`J` is Jain's fairness index over per-replica dispatch counts. "
        "TTFT is from scheduled arrival (open loop); `disp.` columns are from actual "
        "dispatch, and `lag p95` is the client's own dispatch lag. `lagged` counts "
        "repetitions flagged for client lag (kept, see load.py). Past saturation, "
        "latency describes admitted requests only; read it with `err`.",
        "",
        "| policy | ratio | rate (req/s) | N | lagged | TTFT p50 (s) | TTFT p95 (s) "
        "| disp. TTFT p50 (s) | disp. TTFT p95 (s) | lag p95 (s) | e2e p50 (s) "
        "| achieved (req/s) | err | hit rate | J | peak/mean |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(s: tuple[float, float, float] | None, nd: int = 3) -> str:
        if s is None:
            return "not measured"
        return f"{s[0]:.{nd}f} ({s[1]:.{nd}f}-{s[2]:.{nd}f})"

    for key in sorted(g, key=lambda k: (POLICY_ORDER.index(k[0]) if k[0] in POLICY_ORDER else 99, k[1], k[2])):
        pol, ratio, rate = key
        reps = g[key]
        row = [
            POLICY_LABEL.get(pol, pol),
            f"{ratio:g}",
            f"{rate:g}",
            str(len(reps)),
            str(sum(1 for p in reps if p.client_lagged)),
            fmt(med_lo_hi([p.ttft("p50") for p in reps])),
            fmt(med_lo_hi([p.ttft("p95") for p in reps])),
            fmt(med_lo_hi([p.ttft_from_dispatch("p50") for p in reps])),
            fmt(med_lo_hi([p.ttft_from_dispatch("p95") for p in reps])),
            fmt(med_lo_hi([p.send_delay("p95") for p in reps])),
            fmt(med_lo_hi([p.e2e("p50") for p in reps]), 2),
            fmt(med_lo_hi([p.throughput_req_s for p in reps]), 2),
            fmt(med_lo_hi([p.error_rate for p in reps]), 3),
            fmt(med_lo_hi([p.fleet_prefix_hit_rate for p in reps]), 3),
            fmt(med_lo_hi([p.jains_index for p in reps]), 3),
            fmt(med_lo_hi([p.max_over_mean for p in reps]), 2),
        ]
        out.append("| " + " | ".join(row) + " |")

    out += ["", "## Policy comparison against round robin", "",
            "Ratio of each policy's median TTFT p50 to round robin's at the same "
            "(ratio, rate). Below 1.0 means faster than the baseline.", "",
            "| ratio | rate (req/s) | least loaded | prefix affinity | prefix affinity (bounded) |",
            "|---|---|---|---|---|"]
    for ratio in sorted({k[1] for k in g}):
        for rate in sorted({k[2] for k in g if k[1] == ratio}):
            base = med_lo_hi([p.ttft("p50") for p in g.get(("round_robin", ratio, rate), [])])
            if base is None or base[0] <= 0:
                continue
            cells = []
            for pol in POLICY_ORDER[1:]:
                s = med_lo_hi([p.ttft("p50") for p in g.get((pol, ratio, rate), [])])
                cells.append("not measured" if s is None else f"{s[0] / base[0]:.2f}x")
            out.append(f"| {ratio:g} | {rate:g} | " + " | ".join(cells) + " |")

    out += ["", "## Per-replica load and cache hit rate", "",
            "Median over repetitions of each replica's share of dispatched requests "
            "and its own prefix cache hit rate. This is where a hot prefix pinned to "
            "one replica shows up directly.", "",
            "| policy | ratio | rate (req/s) | dispatch share r0 / r1 / r2 / r3 "
            "| hit rate r0 / r1 / r2 / r3 |",
            "|---|---|---|---|---|"]
    for key in sorted(g, key=lambda k: (POLICY_ORDER.index(k[0]) if k[0] in POLICY_ORDER else 99, k[1], k[2])):
        pol, ratio, rate = key
        reps = g[key]
        rids = sorted({r for p in reps for r in p.summary.get("per_replica_dispatch_counts", {})})
        shares, hits = [], []
        for rid in rids:
            sh = []
            for p in reps:
                c = p.summary.get("per_replica_dispatch_counts", {})
                tot = sum(c.values())
                if tot:
                    sh.append(c.get(rid, 0) / tot)
            m = med_lo_hi(sh)
            shares.append("n/m" if m is None else f"{m[0]:.2f}")
            m = med_lo_hi([p.per_replica_hit_rates.get(rid) for p in reps])
            hits.append("n/m" if m is None else f"{m[0]:.2f}")
        out.append(
            f"| {POLICY_LABEL.get(pol, pol)} | {ratio:g} | {rate:g} | "
            + " / ".join(shares) + " | " + " / ".join(hits) + " |"
        )
    out.append("")
    out.append("`n/m` = not measured for that replica in these repetitions.")
    out.append("")
    return out


def main() -> int:
    global SUFFIX
    FIGS.mkdir(parents=True, exist_ok=True)
    trials, finv = load_failure_trials()
    print("Failure-mode inventory:", json.dumps(finv))

    groups = prefix_group_counts() or [1]
    body: list[str] = []
    fleet_caption = "no measurements"
    for g in groups:
        phases, inventory = load_measurements(num_prefix_groups=g)
        caption = hardware_caption(phases)
        if g == groups[0]:
            fleet_caption = caption
        SUFFIX = "" if g == 1 else f"_groups{g}"
        print(f"\nExperiment: {g} prefix group(s)")
        print(json.dumps({k: v for k, v in inventory.items() if k != "degraded_detail"}, indent=2))
        if phases:
            print(f"Generating figures for {len(phases)} usable phases...")
            fig_ttft_vs_rate(phases, caption)
            fig_fairness(phases, caption)
            fig_prefix_cache(phases, caption)
            fig_throughput_error(phases, caption)
            fig_spill(phases, caption)
            fig_streaming_evidence(phases, caption)
        title = (
            "Matrix: one shared prefix (a single hot system prompt)"
            if g == 1
            else f"Matrix: {g} distinct shared prefixes (tenant mix)"
        )
        sec = results_section(phases, inventory, caption)
        body += [f"## {title}", ""]
        body += ["#" + line if line.startswith("##") else line for line in sec]
        if g != 1:
            body.insert(len(body) - len(sec), f"Figures carry the suffix `{SUFFIX}`.\n")

    SUFFIX = ""
    fig_failure(trials, fleet_caption)
    out = [
        "# Results",
        "",
        "Generated by `analysis/make_figures.py` from `results/raw/` only.",
        "Do not edit by hand: rerun `sbatch slurm/run_analysis.sh`.",
        "",
        *body,
        *failure_md(trials, finv),
    ]
    (REPO / "results" / "RESULTS.md").write_text("\n".join(out))
    print(f"  wrote results/RESULTS.md ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
