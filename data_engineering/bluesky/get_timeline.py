import os

from dotenv import load_dotenv
from atproto import Client

load_dotenv()

client = Client()
client.login(os.environ['DUCKDB_ID'], os.environ['DUCKDB_PASSWORD'])

data = client.get_timeline(cursor='', limit=30)

feed = data.feed
print(feed)

next_page = data.cursor
print(next_page)