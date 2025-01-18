INSERT INTO prices
VALUES(1, 11.59, '2018-12-01', '2019-01-01');

INSERT INTO prices
VALUES (1, 11.59, '2019-12-01', '2019-01-01')
ON CONFLICT DO NOTHING;

INSERT INTO prices(value, valid_from, valid_until)
VALUES (11.47, '2019-01-01', '2019-02-01'),
       (11.35, '2019-02-01', '2019-03-01'),
       (11.23, '2019-03-01', '2019-04-01'),
       (11.11, '2019-04-01', '2019-05-01'),
       (10.95, '2019-05-01', '2019-06-01');

INSERT INTO prices(value, valid_from, valid_until)
SELECT * FROM 'prices.csv' src;

UPDATE prices
SET valid_until = valid_from + INTERVAL 1 MONTH
WHERE valid_until IS NULL;