# Source and provenance: Olist

## Original data

Brazilian e-commerce data published by **Olist** on Kaggle (real, anonymized,
2016 to 2018), two datasets combined:

- **Brazilian E-Commerce Public Dataset by Olist** — https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
  (9 marketplace tables: orders, order items, products, sellers, customers, payments, reviews, geolocation, category translation)
- **Marketing Funnel by Olist** — https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist
  (2 tables: closed deals, marketing qualified leads; join to the marketplace via `seller_id`)

The CSVs in `data/` are these public datasets loaded **as-is** (no reshaping).
They match the ontology table and column names one-to-one; the Kaggle links are
the upstream source of truth.

## Notes that keep framing honest

- Store and partner names in review text were replaced with Game of Thrones
  great-house names by the dataset author.
- Today's olist.com (2026) is an ERP/SaaS product, not the marketplace this data
  describes. Frame to the dataset (a marketplace), not to current Olist.
- Amounts are BRL in source; the ontology treats them as EUR.

## Tables (11)

`sellers`, `products`, `product_category_name_translation`, `customers`, `orders`,
`order_items`, `order_payments`, `order_reviews`, `geolocation` (~1M rows, the
heavy file), `marketing_qualified_leads`, `closed_deals`.

## License

The upstream Kaggle datasets are distributed under a **non-commercial Creative
Commons license (CC BY-NC-SA 4.0)** — see the Kaggle dataset pages. The prepared
copies fetched by `datasets/download.py` are redistributed under the same terms,
for research and benchmarking use only.
