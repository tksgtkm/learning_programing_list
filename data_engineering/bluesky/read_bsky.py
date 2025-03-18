import os

import duckdb
from dotenv import load_dotenv
from atproto import Client

con = duckdb.connect(database=':memory:')

# query = """
# SELECT * FROM read_json_auto('https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle?handle=ssp.sh');
# """

# print(con.execute(query).fetchall())

# query = """
# INSTALL http_client FROM community;
# LOAD http_client;
# WITH __input AS (
#     SELECT
#         http_get('https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle?handle=ssp.sh') AS res
# )
# SELECT
#     res::json->>'body' as identity_json
# FROM
#     __input;
# """

# df = con.execute(query).fetch_df()

query = """ \
SET variable did_value = 'did:plc:edglm4muiyzty2snc55ysuqx';

CREATE MACRO get_engagement_data(did_value) AS TABLE (
    WITH raw_data AS (
        SELECT * FROM read_json_auto(
            'https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?actor=' || did_value || '&limit=100'
        )
    ),
    unnested_feed AS (
        SELECT unnest(feed) AS post_data FROM raw_data
    ),
    engagement_data AS (
        SELECT
            RIGHT(post_data.post.uri, 13) AS post_uri,
            post_data.post.author.handle,
            LEFT(post_data.post.record.text, 50) AS post_text,
            post_data.post.record.createdAt AS created_at,
            (post_data.post.replyCount +
             post_data.post.repostCount +
             post_data.post.likeCount +
             post_data.post.quoteCount) AS total_engagement,
            post_data.post.replyCount AS replies,
            post_data.post.repostCount AS reposts,
            post_data.post.likeCount AS likes,
            post_data.post.quoteCount AS quotes
        FROM
            unnested_feed
    )
    SELECT
        post_uri,
        created_at,
        total_engagement,
        bar(
            total_engagement, 0,
            (SELECT MAX(total_engagement) FROM engagement_data),
            30
        ) AS engagement_chart,
        replies, reposts, likes, quotes, post_text
    FROM
        engagement_data
    ORDER BY
        total_engagement DESC
    LIMIT
        30
);

SELECT * FROM get_engagement_data(getvariable('did_value'));
"""

print(con.execute(query).fetchall())