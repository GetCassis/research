# Source and provenance: theLook e-commerce

## Original data

theLook is a synthetic dataset published by Google as a **BigQuery public
dataset** (generated, not real customer data):

- `bigquery-public-data.thelook_ecommerce` — https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce

The CSVs in `data/` are the loaded tables (UPPERCASE column names, matching the
ontology objects).

## What's bundled (6 of 7 tables)

`distribution_centers` (10), `users` (100,000), `products` (29,120),
`orders` (124,923), `order_items` (180,952), `inventory_items` (487,901).

**`events` is not bundled.** The web-events table is 2.4M rows (~143MB gzipped),
too heavy to ship. The benchmark does not use it; if you need clickstream, pull
`events` directly from the BigQuery public dataset above.

## License

Synthetic data published through Google's BigQuery public dataset program
(CC BY 4.0 — see the listing above).
