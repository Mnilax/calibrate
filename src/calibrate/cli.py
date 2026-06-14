"""CLI interface for the calibration tracker."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from calibrate import storage
from calibrate.scoring import (
    base_rate,
    brier_score,
    calibration_bins,
    log_loss,
    murphy_decomposition,
)
from calibrate.plot import plot_calibration_curve

app = typer.Typer(
    name="calibrate",
    help="Forecast calibration tracker — Brier score, log loss, calibration curve.",
    no_args_is_help=True,
)
console = Console()

data_option = typer.Option(None, "--data", help="Override path to data JSON file")


@app.command()
def add(
    question: str = typer.Argument(..., help="The forecast question"),
    probability: float = typer.Argument(..., help="Your P(YES), between 0 and 1"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Category tag"),
    data: Optional[str] = data_option,
) -> None:
    """Add a new prediction."""
    if not 0 < probability < 1:
        console.print("[red]Error:[/red] probability must be between 0 and 1 (exclusive)")
        raise typer.Exit(1)

    store = storage.load(data)
    pred = store.add(question, probability, category)
    storage.save(store, data)

    console.print(f"[green]Added[/green] prediction #{pred.id}: {pred.question} (p={pred.probability})")


@app.command("list")
def list_predictions(
    open: bool = typer.Option(False, "--open", help="Show only open predictions"),
    resolved: bool = typer.Option(False, "--resolved", help="Show only resolved predictions"),
    all: bool = typer.Option(False, "--all", help="Show all predictions"),
    data: Optional[str] = data_option,
) -> None:
    """List predictions."""
    store = storage.load(data)

    if resolved:
        preds = store.resolved_predictions()
        title = "Resolved Predictions"
    elif all:
        preds = store.predictions
        title = "All Predictions"
    else:
        preds = store.open_predictions()
        title = "Open Predictions"

    if not preds:
        console.print(f"[dim]No {title.lower()} found.[/dim]")
        return

    table = Table(title=title, show_lines=False)
    table.add_column("ID", style="bold", width=4)
    table.add_column("Question", max_width=50)
    table.add_column("P(YES)", justify="right", width=7)
    table.add_column("Category", width=10)
    table.add_column("Status", width=10)
    table.add_column("Outcome", width=8)
    table.add_column("Created", width=12)

    for p in preds:
        status = "[green]resolved[/green]" if p.resolved else "[yellow]open[/yellow]"
        outcome = ""
        if p.outcome is not None:
            outcome = "[green]YES[/green]" if p.outcome else "[red]NO[/red]"
        created = p.created_at[:10] if p.created_at else ""
        table.add_row(
            str(p.id),
            p.question,
            f"{p.probability:.2f}",
            p.category or "",
            status,
            outcome,
            created,
        )

    console.print(table)


@app.command()
def resolve(
    pred_id: int = typer.Argument(..., help="Prediction ID to resolve"),
    outcome: str = typer.Argument(..., help="Outcome: yes or no"),
    data: Optional[str] = data_option,
) -> None:
    """Resolve a prediction with its outcome."""
    outcome_lower = outcome.lower()
    if outcome_lower not in ("yes", "no"):
        console.print("[red]Error:[/red] outcome must be 'yes' or 'no'")
        raise typer.Exit(1)

    store = storage.load(data)
    pred = store.get(pred_id)

    if pred is None:
        console.print(f"[red]Error:[/red] prediction #{pred_id} not found")
        raise typer.Exit(1)

    if pred.resolved:
        console.print(f"[yellow]Warning:[/yellow] prediction #{pred_id} already resolved")
        raise typer.Exit(1)

    pred.resolved = True
    pred.outcome = outcome_lower == "yes"
    pred.resolved_at = datetime.now(timezone.utc).isoformat()
    storage.save(store, data)

    result = "[green]YES[/green]" if pred.outcome else "[red]NO[/red]"
    console.print(f"Resolved #{pred_id}: {pred.question} -> {result}")


@app.command()
def score(
    by_category: bool = typer.Option(False, "--by-category", help="Break down by category"),
    data: Optional[str] = data_option,
) -> None:
    """Compute Brier score, log loss, and Murphy decomposition."""
    store = storage.load(data)
    resolved = store.resolved_predictions()

    if not resolved:
        console.print("[dim]No resolved predictions to score.[/dim]")
        return

    def print_scores(preds, label: str = "") -> None:
        pairs = [(p.probability, int(p.outcome)) for p in preds]
        bs = brier_score(pairs)
        ll = log_loss(pairs)
        br = base_rate(pairs)
        murphy = murphy_decomposition(pairs)

        if label:
            console.print(f"\n[bold]{label}[/bold]")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Predictions", str(len(preds)))
        table.add_row("Base rate (YES freq)", f"{br:.4f}")
        table.add_row("Brier score", f"{bs:.6f}")
        table.add_row("Log loss", f"{ll:.6f}")
        table.add_row("", "")
        table.add_row("[dim]Murphy decomposition[/dim]", "")
        table.add_row("  Reliability", f"{murphy.reliability:.6f}")
        table.add_row("  Resolution", f"{murphy.resolution:.6f}")
        table.add_row("  Uncertainty", f"{murphy.uncertainty:.6f}")
        table.add_row("", "")
        table.add_row(
            "  Check: Rel - Res + Unc",
            f"{murphy.brier_check:.6f}",
        )
        table.add_row("  Brier score", f"{bs:.6f}")

        delta = abs(murphy.brier_check - bs)
        check_status = "[green]✓ match[/green]" if delta < 0.01 else f"[yellow]Δ = {delta:.4f}[/yellow]"
        table.add_row("  Identity check", check_status)

        console.print(table)

    print_scores(resolved, "Overall")

    if by_category:
        categories: dict[str, list] = {}
        for p in resolved:
            cat = p.category or "uncategorized"
            categories.setdefault(cat, []).append(p)
        for cat, preds in sorted(categories.items()):
            print_scores(preds, f"Category: {cat}")


@app.command()
def plot(
    out: str = typer.Option("assets/calibration.png", "--out", help="Output PNG path"),
    bins: int = typer.Option(10, "--bins", help="Number of calibration bins"),
    data: Optional[str] = data_option,
) -> None:
    """Generate calibration curve PNG."""
    store = storage.load(data)
    resolved = store.resolved_predictions()

    if not resolved:
        console.print("[dim]No resolved predictions to plot.[/dim]")
        return

    pairs = [(p.probability, int(p.outcome)) for p in resolved]
    output_path = plot_calibration_curve(pairs, out, bins)
    console.print(f"[green]Saved[/green] calibration curve to {output_path}")


@app.command()
def seed(
    data: Optional[str] = data_option,
) -> None:
    """Load sample predictions from examples/sample_predictions.json."""
    # Find the examples file relative to package
    examples_paths = [
        Path(__file__).parent.parent.parent / "examples" / "sample_predictions.json",
        Path.cwd() / "examples" / "sample_predictions.json",
    ]

    sample_path = None
    for p in examples_paths:
        if p.exists():
            sample_path = p
            break

    if sample_path is None:
        console.print("[red]Error:[/red] examples/sample_predictions.json not found")
        raise typer.Exit(1)

    with open(sample_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    from calibrate.models import PredictionStore

    store = PredictionStore.from_dict(sample_data)
    storage.save(store, data)

    n_resolved = len(store.resolved_predictions())
    n_open = len(store.open_predictions())
    console.print(
        f"[green]Loaded[/green] {len(store.predictions)} predictions "
        f"({n_resolved} resolved, {n_open} open)"
    )


if __name__ == "__main__":
    app()
