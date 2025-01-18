import duckdb

con = duckdb.connect(database='first.db')

con.execute('SELECT * FROM readings')

print(con.fetchall())