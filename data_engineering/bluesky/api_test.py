import os
from dotenv import load_dotenv
from atproto import Client

load_dotenv()

client = Client()
client.login(os.environ['DUCKDB_ID'], os.environ['DUCKDB_PASSWORD'])

post = client.send_post('こんにちは、これはSDKで投稿しています')

print('===== post.uri =====')
print(post.uri)
print('===== post.cid =====')
print(post.cid)