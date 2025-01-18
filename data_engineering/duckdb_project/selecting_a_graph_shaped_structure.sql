CREATE TABLE IF NOT EXISTS src (
    id INT PRIMARY KEY,
    parent_id INT,
    name VARCHAR(8)
);

INSERT INTO src (
    VALUES
    (1, null, 'root1'),
    (2, 1, 'ch1a'),
    (3, 1, 'ch2a'),
    (4, 3, 'ch3a'),
    (5, null, 'root2'),
    (6, 5, 'ch1b')
);

WITH RECURSIVE tree AS (
    SELECT
        id,
        id AS root_id,
        [name] AS path
    FROM
        src
    WHERE
        parent_id IS NULL
    UNION ALL
    SELECT
        src.id,
        root_id,
        list_append(tree.path, src.name) AS path
    FROM
        src
    JOIN tree ON (src.parent_id = tree.id)
)
SELECT
    path
FROM
    tree;