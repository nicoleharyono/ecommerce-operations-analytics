WITH review_by_order AS (
    SELECT
        order_id,
        AVG(review_score) AS review_score
    FROM olist_order_reviews_dataset
    GROUP BY order_id
),

delivery_performance AS (
    SELECT
        order_id,
        CASE
            WHEN date(order_delivered_customer_date)
                 > date(order_estimated_delivery_date)
            THEN 1
            ELSE 0
        END AS is_late
    FROM olist_orders_dataset
    WHERE order_delivered_customer_date IS NOT NULL
)

SELECT
    d.is_late,
    COUNT(*) AS order_count,
    AVG(r.review_score) AS average_review_score
FROM delivery_performance AS d
LEFT JOIN review_by_order AS r
    ON d.order_id = r.order_id
GROUP BY d.is_late
ORDER BY d.is_late;

----

WITH delivery_performance AS (
    SELECT
        order_id,
        CASE
            WHEN date(order_delivered_customer_date)
                 > date(order_estimated_delivery_date)
            THEN 1
            ELSE 0
        END AS is_late,

        CAST(
            julianday(order_delivered_carrier_date)
            - julianday(order_approved_at)
            AS INTEGER
        ) AS handling_days

    FROM olist_orders_dataset
    WHERE order_delivered_customer_date IS NOT NULL
      AND order_delivered_carrier_date IS NOT NULL
      AND order_approved_at IS NOT NULL
),

valid_orders AS (
    SELECT
        order_id,
        is_late,
        handling_days,
        CASE
            WHEN handling_days <= 1 THEN '0–1 days'
            WHEN handling_days <= 3 THEN '2–3 days'
            WHEN handling_days <= 7 THEN '4–7 days'
            ELSE '8+ days'
        END AS handling_group
    FROM delivery_performance
    WHERE handling_days >= 0
)

SELECT
    handling_group,
    COUNT(*) AS order_count,
    ROUND(AVG(is_late) * 100, 2) AS late_delivery_rate_pct
FROM valid_orders
GROUP BY handling_group
ORDER BY
    CASE handling_group
        WHEN '0–1 days' THEN 1
        WHEN '2–3 days' THEN 2
        WHEN '4–7 days' THEN 3
        WHEN '8+ days' THEN 4
    END;

-----

WITH delivery_performance AS (
    SELECT
        order_id,
        CASE
            WHEN date(order_delivered_customer_date)
                 > date(order_estimated_delivery_date)
            THEN 1
            ELSE 0
        END AS is_late,

        CAST(
            julianday(order_delivered_carrier_date)
            - julianday(order_approved_at)
            AS INTEGER
        ) AS handling_days

    FROM olist_orders_dataset
    WHERE order_delivered_customer_date IS NOT NULL
),

seller_orders AS (
    SELECT DISTINCT
        i.order_id,
        i.seller_id,
        d.is_late,
        d.handling_days
    FROM olist_order_items_dataset AS i
    INNER JOIN delivery_performance AS d
        ON i.order_id = d.order_id
),

seller_performance AS (
    SELECT
        seller_id,
        COUNT(DISTINCT order_id) AS order_count,
        SUM(is_late) AS late_order_count,
        AVG(is_late) * 100 AS late_delivery_rate_pct,
        AVG(handling_days) AS average_handling_days
    FROM seller_orders
    GROUP BY seller_id
)

SELECT
    seller_id,
    order_count,
    late_order_count,
    ROUND(late_delivery_rate_pct, 2) AS late_delivery_rate_pct,
    ROUND(average_handling_days, 2) AS average_handling_days
FROM seller_performance
WHERE order_count >= 30
ORDER BY late_order_count DESC
LIMIT 10;


----

WITH delivery_performance AS (
    SELECT
        order_id,
        CASE
            WHEN date(order_delivered_customer_date)
                 > date(order_estimated_delivery_date)
            THEN 1
            ELSE 0
        END AS is_late
    FROM olist_orders_dataset
    WHERE order_delivered_customer_date IS NOT NULL
),

order_categories AS (
    SELECT DISTINCT
        i.order_id,
        COALESCE(
            t.product_category_name_english,
            p.product_category_name
        ) AS product_category_name_english
    FROM olist_order_items_dataset AS i
    INNER JOIN olist_products_dataset AS p
        ON i.product_id = p.product_id
    LEFT JOIN product_category_name_translation AS t
        ON p.product_category_name = t.product_category_name
),

category_performance AS (
    SELECT
        c.product_category_name_english,
        COUNT(DISTINCT c.order_id) AS order_count,
        SUM(d.is_late) AS late_order_count,
        AVG(d.is_late) * 100 AS late_delivery_rate_pct
    FROM order_categories AS c
    INNER JOIN delivery_performance AS d
        ON c.order_id = d.order_id
    WHERE c.product_category_name_english IS NOT NULL
    GROUP BY c.product_category_name_english
)

SELECT
    product_category_name_english,
    order_count,
    late_order_count,
    ROUND(late_delivery_rate_pct, 2) AS late_delivery_rate_pct
FROM category_performance
WHERE order_count >= 100
ORDER BY late_order_count DESC
LIMIT 10;

conn.close()