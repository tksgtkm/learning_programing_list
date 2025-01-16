import duckdb

con = duckdb.connect(database=':memory:')

con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")

population = con.read_csv('https://bit.ly/3KoiZR0')

print(type(population))

print(con.execute("select * from population limit 2").fetchall())

population.to_table("population")

population_table = con.table("population")

print(population_table)

print(population_table.count("*").show())

print(
    (population_table
     .filter("Population > 10000000")
     .project("Country, Population")
     .limit(5)
     .show()
     )
)

over_10m = population_table.filter('Population > 10000000')

print(
    (over_10m
     .aggregate("Region, CAST(avg(Population) as int) as pop")
     .order("pop DESC"))
)

print(
    over_10m
    .filter('"GDP ($ per capita)" > 10000')
    .count("*")
)

