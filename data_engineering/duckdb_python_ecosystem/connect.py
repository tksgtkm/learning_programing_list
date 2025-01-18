import duckdb

con = duckdb.connect(database='../duckdb_project/first.db')

con.execute('SELECT * FROM readings')

print(con.fetchall())