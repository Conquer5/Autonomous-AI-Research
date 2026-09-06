from __future__ import annotations

from datetime import date

import httpx

from research_radar.tools.news import RssNewsTool

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Efficient Tech</title>
  <item>
    <title>Free local AI gets faster on CPU laptops</title>
    <guid isPermaLink="true">https://example.com/local-ai</guid>
    <description><![CDATA[An <b>open-source</b> efficient inference release.]]></description>
    <pubDate>Thu, 27 Aug 2026 10:00:00 GMT</pubDate>
  </item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Engineering Blog</title>
  <entry>
    <title>Gardening update</title>
    <link href="https://example.com/garden" />
    <summary>Plants and soil.</summary>
    <updated>2026-08-27T10:00:00Z</updated>
  </entry>
</feed>"""


async def test_rss_news_supports_guid_filters_topics_and_partial_feeds() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rss":
            return httpx.Response(200, content=RSS)
        if request.url.path == "/atom":
            return httpx.Response(200, content=ATOM)
        return httpx.Response(503)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = RssNewsTool(
        ("https://example.com/rss", "https://example.com/atom", "https://example.com/down"),
        client=http,
    )

    result = await tool.search(
        ("efficient local AI", "free open-source tools"),
        published_after=date(2026, 8, 1),
    )

    assert [item.title for item in result.items] == ["Free local AI gets faster on CPU laptops"]
    assert str(result.items[0].url) == "https://example.com/local-ai"
    assert result.items[0].description == "An open-source efficient inference release."
    assert result.partial is True
    assert len(result.warnings) == 1
    await http.aclose()


async def test_model_announcement_matches_versioned_gpt_and_unknown_model_name() -> None:
    from research_radar.config import DEFAULT_NEWS_QUERIES

    payload = b"""<rss><channel><title>Model Lab</title>
    <item><title>Introducing GPT-99</title><link>https://example.com/gpt</link>
    <pubDate>Fri, 04 Sep 2026 10:00:00 GMT</pubDate></item>
    <item><title>Introducing Fable</title><link>https://example.com/fable</link>
    <description>A new reasoning model is available today.</description>
    <pubDate>Fri, 04 Sep 2026 11:00:00 GMT</pubDate></item>
    <item><title>Gardening</title><link>https://example.com/garden</link>
    <description>Plants and soil.</description></item>
    </channel></rss>"""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
    ) as client:
        tool = RssNewsTool(("https://example.com/feed",), client=client)
        result = await tool.search(DEFAULT_NEWS_QUERIES)
    assert [item.title for item in result.items] == ["Introducing Fable", "Introducing GPT-99"]


def test_news_interleaves_publishers_and_prioritizes_recent_articles() -> None:
    from datetime import UTC, datetime

    from research_radar.schemas import NewsResult
    from research_radar.tools.news import diverse_news

    items = [
        NewsResult(
            title="Older keyword-heavy post",
            url="https://a.test/old",
            published_at=datetime(2026, 9, 1, tzinfo=UTC),
            relevance_score=1,
        ),
        NewsResult(
            title="New release",
            url="https://a.test/new",
            published_at=datetime(2026, 9, 4, tzinfo=UTC),
            relevance_score=0.3,
        ),
        NewsResult(
            title="Other lab",
            url="https://b.test/model",
            published_at=datetime(2026, 9, 3, tzinfo=UTC),
            relevance_score=0.4,
        ),
    ]
    assert [item.title for item in diverse_news(items)] == [
        "New release",
        "Other lab",
        "Older keyword-heavy post",
    ]


def test_aggregated_news_preserves_publisher_for_diversity() -> None:
    from research_radar.tools.news import diverse_news

    payload = b"""<rss><channel><title>Google News</title>
    <item><title>Model A</title><link>https://news.google.com/rss/articles/a</link>
    <source url="https://a.test">Publisher A</source></item>
    <item><title>Model B</title><link>https://news.google.com/rss/articles/b</link>
    <source url="https://a.test">Publisher A</source></item>
    <item><title>Model C</title><link>https://news.google.com/rss/articles/c</link>
    <source url="https://b.test">Publisher B</source></item>
    </channel></rss>"""
    items = RssNewsTool._parse_feed(
        payload, "https://news.google.com/rss/search", tokens={"model"}, published_after=None
    )
    assert [item.source_name for item in diverse_news(items)] == [
        "Publisher A",
        "Publisher B",
        "Publisher A",
    ]
