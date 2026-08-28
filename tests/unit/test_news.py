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
