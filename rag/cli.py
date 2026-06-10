import click
from dotenv import load_dotenv
from rag.factory import build_pipeline
from rag.ingest import parse_posts

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
