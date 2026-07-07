import json
import random
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rag.factory import (build_pipeline, build_retriever, build_all_retrievers,
                         build_dense_and_bm25, build_llm, build_judge_llm)
from rag.retrieve import HybridRetriever
from rag.ingest import parse_posts
from rag.eval.testset import build_testset, save_testset, load_testset
from rag.eval.runner import run_eval
from rag.generate import Generator
from rag.eval.quality import LLMJudge, run_quality_eval

load_dotenv()


@click.group()
def cli():
    """Cooking RAG command-line interface."""


@cli.command()
@click.argument("posts_xml", type=click.Path(exists=True))
def index(posts_xml):
    """Parse Posts.xml and build the vector index."""
    pipe = build_pipeline()
    docs = parse_posts(posts_xml)
    answers = sum(1 for d in docs if d.question_id != d.post_id)
    pipe.index(docs)
    click.echo(
        f"Indexed {pipe.store.count()} chunks from {answers} answers "
        f"({len(docs)} posts parsed)."
    )


@cli.command()
@click.argument("question")
@click.option("-k", default=5, help="Number of passages to retrieve.")
def ask(question, k):
    """Ask a cooking question and get a grounded, cited answer."""
    pipe = build_pipeline()
    answer = pipe.query(question, k=k)
    click.echo(answer.text)
    if answer.cited_post_ids:
        click.echo("\nSources:")
        for pid in sorted(set(answer.cited_post_ids)):
            click.echo(f"  https://cooking.stackexchange.com/a/{pid}")


@cli.command("eval-build")
@click.argument("posts_xml", type=click.Path(exists=True))
@click.option("--n", default=500, help="Number of test cases to sample.")
@click.option("--seed", default=42, help="Random seed for sampling.")
@click.option("--out", default="data/eval/testset.json", help="Output JSON path.")
def eval_build(posts_xml, n, seed, out):
    """Build a held-out retrieval test set from Posts.xml."""
    docs = parse_posts(posts_xml)
    cases = build_testset(docs, n=n, seed=seed)
    save_testset(cases, out)
    noun = "test case" if len(cases) == 1 else "test cases"
    click.echo(f"Wrote {len(cases)} {noun} to {out}.")


@cli.command("eval")
@click.option("--testset", "testset_path", default="data/eval/testset.json",
              type=click.Path(exists=True), help="Test set JSON path.")
@click.option("--retriever", "kind",
              type=click.Choice(["dense", "bm25", "hybrid", "rerank"]), default="dense",
              help="Retriever to evaluate.")
@click.option("--fetch", default=50, help="Chunks to fetch before deduping.")
def eval_cmd(testset_path, kind, fetch):
    """Run Setup 1 retrieval evaluation and print the metrics table."""
    ks = (1, 5, 10)
    if fetch <= max(ks):
        raise click.BadParameter(f"--fetch must exceed {max(ks)}", param_hint="--fetch")
    cases = load_testset(testset_path)
    retriever = build_retriever(kind)
    report = run_eval(cases, retriever, ks=ks, fetch_depth=fetch)
    click.echo(
        f"Setup 1 — retrieval (n={report.n_cases}, retriever={kind}, fetch={fetch})")
    click.echo("Hit@1   Hit@5   Hit@10   MRR")
    click.echo(
        f"{report.hit_at[1]:.2f}    {report.hit_at[5]:.2f}    "
        f"{report.hit_at[10]:.2f}     {report.mrr:.2f}"
    )


@cli.command("eval-compare")
@click.option("--testset", "testset_path", default="data/eval/testset.json",
              type=click.Path(exists=True), help="Test set JSON path.")
@click.option("--fetch", default=50, help="Chunks to fetch before deduping.")
def eval_compare(testset_path, fetch):
    """Run all retrievers over the test set and print a comparison table."""
    ks = (1, 5, 10)
    if fetch <= max(ks):
        raise click.BadParameter(f"--fetch must exceed {max(ks)}", param_hint="--fetch")
    cases = load_testset(testset_path)
    retrievers = build_all_retrievers()
    click.echo(f"Setup 1 — retrieval comparison (n={len(cases)}, fetch={fetch})")
    click.echo("retriever   Hit@1   Hit@5   Hit@10   MRR")
    for name, retriever in retrievers.items():
        report = run_eval(cases, retriever, ks=ks, fetch_depth=fetch)
        click.echo(
            f"{name:<10}  {report.hit_at[1]:.2f}    {report.hit_at[5]:.2f}    "
            f"{report.hit_at[10]:.2f}     {report.mrr:.2f}"
        )


@cli.command("eval-sweep")
@click.option("--testset", "testset_path", default="data/eval/testset.json",
              type=click.Path(exists=True), help="Test set JSON path.")
@click.option("--weights", default="0,0.25,0.5,1.0",
              help="Comma-separated BM25 weights to sweep.")
@click.option("--fetch", default=50, help="Chunks to fetch before deduping.")
def eval_sweep(testset_path, weights, fetch):
    """Sweep BM25 fusion weights for the hybrid retriever and print a table."""
    ks = (1, 5, 10)
    if fetch <= max(ks):
        raise click.BadParameter(f"--fetch must exceed {max(ks)}", param_hint="--fetch")
    weight_values = [float(w) for w in weights.split(",")]
    cases = load_testset(testset_path)
    dense, sparse = build_dense_and_bm25()
    click.echo(f"Setup 1 — hybrid weight sweep (n={len(cases)}, fetch={fetch})")
    click.echo("bm25_weight   Hit@1   Hit@5   Hit@10   MRR")
    for w in weight_values:
        retriever = HybridRetriever([dense, sparse], weights=[1.0, w])
        report = run_eval(cases, retriever, ks=ks, fetch_depth=fetch)
        click.echo(
            f"{w:<11}   {report.hit_at[1]:.2f}    {report.hit_at[5]:.2f}    "
            f"{report.hit_at[10]:.2f}     {report.mrr:.2f}"
        )


@cli.command("eval-quality")
@click.option("--testset", "testset_path", default="data/eval/testset.json",
              type=click.Path(exists=True), help="Test set JSON path.")
@click.option("--posts", "posts_xml", default="data/Posts.xml",
              type=click.Path(exists=True), help="Posts.xml for accepted-answer text.")
@click.option("--retriever", "kind",
              type=click.Choice(["dense", "bm25", "hybrid", "rerank"]), default="dense",
              help="Retriever to evaluate.")
@click.option("--n", default=50, help="Number of cases to sample from the test set.")
@click.option("--seed", default=42, help="Random seed for sampling.")
@click.option("-k", "k", default=5, help="Passages fed to the generator.")
@click.option("--out", default=None, help="Artifact JSON path (default auto-named).")
def eval_quality(testset_path, posts_xml, kind, n, seed, k, out):
    """Generate answers and LLM-judge them for correctness + faithfulness."""
    cases = load_testset(testset_path)
    sample = random.Random(seed).sample(cases, min(n, len(cases)))
    docs = parse_posts(posts_xml)
    reference_texts = {d.post_id: d.body for d in docs}
    kept = []
    for c in sample:
        if c.accepted_post_id in reference_texts:
            kept.append(c)
        else:
            click.echo(
                f"warning: accepted answer {c.accepted_post_id} for question "
                f"{c.question_id} not found in Posts.xml; skipping", err=True)
    gen_llm = build_llm(temperature=0.0)
    judge_llm = build_judge_llm(temperature=0.0)
    generator = Generator(llm=gen_llm, min_score=0.2)
    judge = LLMJudge(judge_llm)
    retriever = build_retriever(kind)
    report, results = run_quality_eval(
        kept, retriever, generator, judge, reference_texts, k=k)
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = f"data/eval/quality/{kind}-{stamp}.json"
    artifact = {
        "config": {"retriever": kind, "n": len(sample),
                   "n_skipped": len(sample) - len(kept), "seed": seed, "k": k,
                   "model": getattr(gen_llm, "_model", "unknown"),
                   "judge_model": getattr(judge_llm, "_model", "unknown")},
        "summary": {"mean_correctness": report.mean_correctness,
                    "mean_faithfulness": report.mean_faithfulness,
                    "refusal_rate": report.refusal_rate,
                    "n_judged": report.n_judged,
                    "n_judge_errors": report.n_judge_errors},
        "cases": [vars(r) for r in results],
    }
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    refused = report.n_cases - report.n_judged - report.n_judge_errors
    note = f"   [{refused} refused]" if refused else ""
    click.echo(f"Answer quality (n={report.n_judged}, retriever={kind}, k={k}){note}")
    if report.n_judged:
        click.echo("correctness  faithfulness")
        click.echo(
            f"{report.mean_correctness:.2f}         {report.mean_faithfulness:.2f}")
    else:
        click.echo("no cases judged")
    click.echo(f"\nWrote artifact to {out}.")
