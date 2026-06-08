import re
from pathlib import Path
from lxml import etree
from rag.models import Document

_TAG_RE = re.compile(r"<([^>]+)>")
_HTML_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    return _HTML_RE.sub("", html or "").strip()


def _parse_tags(raw: str) -> list[str]:
    return _TAG_RE.findall(raw or "")


def parse_posts(path: Path) -> list[Document]:
    """Parse a Stack Exchange Posts.xml into Documents.

    Questions (PostTypeId=1) and answers (PostTypeId=2) both become Documents.
    An answer is flagged is_accepted if its question's AcceptedAnswerId == its Id.
    """
    tree = etree.parse(str(path))
    rows = tree.findall("row")

    accepted_ids: set[int] = set()
    for row in rows:
        acc = row.get("AcceptedAnswerId")
        if acc is not None:
            accepted_ids.add(int(acc))

    docs: list[Document] = []
    for row in rows:
        post_id = int(row.get("Id"))
        post_type = row.get("PostTypeId")
        if post_type == "1":
            question_id = post_id
        elif post_type == "2":
            question_id = int(row.get("ParentId"))
        else:
            continue

        docs.append(
            Document(
                post_id=post_id,
                question_id=question_id,
                title=row.get("Title", "") or "",
                body=_strip_html(row.get("Body", "")),
                tags=_parse_tags(row.get("Tags", "")),
                score=int(row.get("Score", "0")),
                is_accepted=post_id in accepted_ids,
            )
        )
    return docs
