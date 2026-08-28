from __future__ import annotations

from datetime import date

import httpx

from research_radar.tools.arxiv import ArxivSearchTool
from research_radar.utils.retry import RetryPolicy

ATOM_FEED = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2608.12345v1</id>
    <updated>2026-08-25T10:00:00Z</updated>
    <published>2026-08-24T10:00:00Z</published>
    <title>Reliable Agent Harnesses</title>
    <summary>We evaluate reliable agent harness engineering.</summary>
    <author><name>Alice Example</name></author>
    <category term="cs.AI" />
    <arxiv:primary_category term="cs.AI" />
    <link href="http://arxiv.org/pdf/2608.12345v1" type="application/pdf" />
  </entry>
</feed>"""


async def test_arxiv_search_builds_query_and_parses_atom() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert "cat%3Acs.AI" in str(request.url)
        assert request.url.params["sortBy"] == "submittedDate"
        return httpx.Response(200, content=ATOM_FEED)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = ArxivSearchTool(
        client=http,
        min_interval_seconds=0,
        retry_policy=RetryPolicy(attempts=1),
    )
    result = await tool.search(
        "agent harness", categories=["cs.AI"], published_after=date(2026, 8, 1)
    )

    assert result.total_available == 1
    assert result.items[0].arxiv_id == "2608.12345v1"
    assert result.items[0].authors == ["Alice Example"]
    assert str(result.items[0].url).startswith("https://")
    await http.aclose()
