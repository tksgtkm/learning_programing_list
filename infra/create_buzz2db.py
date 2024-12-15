import duckdb

cursor = duckdb.connect('buzz2.db')
# print(cursor.execute("SELECT 42").fetchall())

cursor.execute("CREATE TABLE t2(i INTEGER, j INTEGER)")
cursor.execute("INSERT INTO t2 VALUES (1, 2), (2, 3), (3, 4)")