import click
from dotenv import load_dotenv
from rag.factory import build_pipeline, build_retriever, build_all_retrievers
from rag.ingest import parse_posts
from rag.eval.testset import build_testset, save_testset, load_testset
from rag.eval.runner import run_eval

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
              type=click.Choice(["dense", "bm25", "hybrid"]), default="dense",
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
