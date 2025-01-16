WITH per_hour AS (
    SELECT
        system_id,
        date_trunc('hour', read_on) AS read_on,
        avg(power) / 1000 AS kWh
    FROM
        readings
    GROUP BY ALL
)
SELECT
    name,
    max(kWh),
    arg_max(read_on, kWh) AS 'Read on'
FROM
    per_hour
JOIN systems s ON s.id = per_hour.system_id
WHERE system_id = 10
GROUP by s.name;