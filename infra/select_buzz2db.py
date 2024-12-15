import duckdb

cursor = duckdb.connect('buzz2.db')

print(cursor.execute("SELECT * FROM t2").fetchall())