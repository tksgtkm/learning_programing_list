WITH max_power AS (
    SELECT max(power) AS v FROM readings
)
SELECT
    max_power.v, read_on
FROM
    max_power
JOIN
    readings
ON
    power = max_power.v;