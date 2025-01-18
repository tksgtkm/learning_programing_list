import duckdb
import locale
import pandas as pd
import polars as pl
import pyarrow.compute as pc

from duckdb.typing import *

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

print(
    (population_table
    .except_(over_10m)
    .aggregate("""
    Region,
    CAST(avg(population) AS int) AS population,
    COUNT(*)
    """))
)

eastern_europe = population_table.filter(
    "Region ~ '.*EASTERN EUROPE.*'"
)

print(
    (eastern_europe
    .intersect(over_10m)
    .project("Country, Population"))
)

# pandas

people = pd.DataFrame({
    "name": ["Michael Hunger", "Michael Simons", "Mark Needham"],
    "country": ["Germany", "Germany", "Great Britain"]
})

print(duckdb.sql(
    "select * from people where country = 'Germany'"
))

params = {"country": "Germany"}
print(duckdb.execute("""
select * from people where country <> $country
""", params).fetchdf())

print(
    con.sql("""
    select distinct Region, length(Region) as numChars from population
    """)
)

def remove_spaces(field:str) -> str:
    if field:
        return field.lstrip().rstrip()
    else:
        return field

con.create_function('remove_spaces', remove_spaces)

print(
    con.sql("""
    select function_name, function_type, parameters, parameter_types, return_type
    from duckdb_functions() where function_name = 'remove_spaces'
    """)
)

print(
    con.sql("select length(remove_spaces('foo'))")
)

# 関数の重複を避けるために削除
con.remove_function('remove_spaces')

con.create_function(
    'remove_spaces',
    remove_spaces,
    [(VARCHAR)],
    VARCHAR
)

print(
    con.sql("""
    select distinct Region, length(Region) as len1,
    remove_spaces(Region) as cleanRegion,
    length (cleanRegion) as len2
    from population
    where len1 between 20 and 30
    limit 3
    """)
)

con.sql("""
update population set Region = remove_spaces(Region)
""")

print(
    con.sql("""
    select distinct Region, length(Region) as numChars 
    from population
    """)
)

def convert_locale(field:str) -> float:
    locale.setlocale(locale.LC_ALL, 'ja_JP.UTF-8')
    return locale.atof(field)

con.create_function('convert_locale', convert_locale)

print(
    con.sql("""
    select "Coastline (coast/area ratio)" as coastline,
    convert_locale(coastline) as cleanCoastline,
    "Pop. Density (per sq. mi.)" as popDen,
    convert_locale(popDen) as cleanPopDen
    from population
    limit 5
    """)
)

population_table = con.table("population")

print(population_table.limit(5).pl()[["Country", "Region", "Population"]])

arrow_table = population_table.to_arrow_table()

print(
    arrow_table
    .filter(pc.field("Region") == "NEAR EAST")
    .select(["Country", "Region", "Population"])
    .slice(length=5)
)