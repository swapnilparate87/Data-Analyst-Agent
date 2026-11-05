# Evaluation: `gemma2:2b` vs `nvidia/nemotron-3.5-lightning-30b-a3b`

- **A**: run `baseline_gemma2-2b`, model `gemma2:2b` (ollama), judge `nvidia/nemotron-3-ultra-550b-a55b`
- **B**: run `nemotron-3.5-lightning-30b`, model `nvidia/nemotron-3.5-lightning-30b-a3b` (nvidia), judge `nvidia/nemotron-3-ultra-550b-a55b`

## Cases passed

| Component | A | B | Change | Time A | Time B |
|---|---|---|---|---|---|
| scope_guard | 20/20 (100%) | 20/20 (100%) | +0 | 71s | 19s |
| router | 18/18 (100%) | 18/18 (100%) | +0 | 104s | 26s |
| entity_resolver | 17/18 (94%) | 17/18 (94%) | +0 | 6s | 4s |
| retriever | 13/18 (72%) | 10/18 (56%) | -3 | 162s | 63s |
| answer_generator | 7/16 (44%) | 11/16 (69%) | +4 | 1505s | 1564s |
| report_generator | 1/15 (7%) | 7/15 (47%) | +6 | 1981s | 6101s |
| **total** | **76/105 (72%)** | **83/105 (79%)** | **+7** | | |

## Metrics (pass rate / average score)

| Component | Metric | A | B | Score change |
|---|---|---|---|---|
| scope_guard | Exact match | 100% / 1.00 | 100% / 1.00 | +0.00 |
| router | Exact match | 100% / 1.00 | 100% / 1.00 | +0.00 |
| entity_resolver | Exact match | 94% / 0.94 | 94% / 0.94 | +0.00 |
| retriever | Result match | 72% / 0.78 | 56% / 0.58 | -0.19 |
| answer_generator | Correctness [GEval] | 56% / 0.56 | 81% / 0.85 | +0.29 |
| answer_generator | Faithfulness | 81% / 0.85 | 88% / 0.89 | +0.04 |
| answer_generator | Answer Relevancy | 94% / 0.91 | 100% / 0.99 | +0.07 |
| report_generator | Numbers grounded | 93% / 0.99 | 80% / 0.98 | -0.01 |
| report_generator | Faithfulness | 93% / 0.91 | 93% / 0.90 | -0.01 |
| report_generator | Report quality [GEval] | 7% / 0.37 | 67% / 0.64 | +0.27 |

## Cases that changed

### entity_resolver
- **Failing in both**: [detect] Compare SP sales with the rest of Brazil
  - A: `city: sales`
  - B: `city: sales`

### retriever
- **Fixed in B**: What was the total revenue in 2017?
  - A: `[{"sum(price)":66650.37},{"sum(price)":27817.58},{"sum(price)":47314.71},{"sum(price)":50912.71},{"sum(price)":5962902.01},{"sum(price)":209.6}]`
  - B: `[{"total_revenue_2017":6108492.27}]`
- **Fixed in B**: What is the average freight value per item?
  - A: `[{"average_freight_per_item":22.25}]`
  - B: `[{"avg_freight_value":19.99}]`
- **Fixed in B**: Which 10 sellers made the most revenue in 2018?
  - A: `[{"seller_id":"4869f7a5dfa277a7dca6462dcf3b52b2","total_revenue":138414.6},{"seller_id":"955fee9216a65b617aa5c0531780ce60","total_revenue":117340.86},{"seller_i…`
  - B: `[{"seller_id":"4869f7a5dfa277a7dca6462dcf3b52b2","total_revenue":138414.6},{"seller_id":"955fee9216a65b617aa5c0531780ce60","total_revenue":117328.86},{"seller_i…`
- **Broken in B**: How many orders were delivered late?
  - A: `[{"count(DISTINCT os.order_id)":7827}]`
  - B: `[{"late_deliveries":8715}]`
- **Broken in B**: How many orders came from Curitiba?
  - A: `[{"count(DISTINCT os.order_id)":1510}]`
  - B: `model error: { "sql": "SELECT COUNT(DISTINCT order_id) AS order_count FROM sales WHERE customer_city = 'curitiba' AND order_status NOT IN ('canceled', 'unavaila…`
- **Broken in B**: What is the average review score in furniture_decor?
  - A: `[{"avg(review_score)":3.91}]`
  - B: `model error: { "sql": "SELECT AVG(review_score) AS avg_review_score FROM sales WHERE category = 'furniture_decor' AND review_score IS NOT NULL; SELECT COUNT(*) …`
- **Broken in B**: How many orders were canceled?
  - A: `[{"count(DISTINCT order_id)":461}]`
  - B: `[{"canceled_orders":244}]`
- **Broken in B**: How many orders came from the state of Rio Grande do Sul?
  - A: `[{"count(DISTINCT os.order_id)":5432}]`
  - B: `model error: { "sql": "SELECT COUNT(DISTINCT order_id) AS order_count FROM sales WHERE customer_state = 'RS' AND order_status NOT IN ('canceled', 'unavailable')…`
- **Broken in B**: How many different products are in the health_beauty category?
  - A: `[{"count(DISTINCT product_id)":2444}]`
  - B: `model error: { "sql": "SELECT COUNT(DISTINCT product_id) AS product_count FROM sales WHERE category = 'health_beauty' AND order_status NOT IN ('canceled', 'unav…`
- **Failing in both**: What is the total freight paid by customers in Belo Horizonte?
  - A: `[{"count(DISTINCT sf.freight_value)":1095}]`
  - B: `model error: { "sql": "SELECT SUM(freight_value) AS total_freight FROM sales WHERE customer_city = 'belo horizonte' AND order_status NOT IN ('canceled', 'unavai…`
- **Failing in both**: What are the 3 states with the highest share of late deliveries?
  - A: `query failed: SELECT state, SUM(CASE WHEN delivered_late = true THEN 1 ELSE 0 END) as late_deliveries_count FROM sales GROUP BY state ORDER BY late_deliveries_c…`
  - B: `[{"customer_state":"AL","total_deliveries":444,"late_deliveries":103.0,"late_delivery_pct":23.2},{"customer_state":"MA","total_deliveries":824,"late_deliveries"…`

### answer_generator
- **Fixed in B**: Which 3 states have the most orders?
  - A: `Here is the result (see the table above).`
  - B: `São Paulo leads with 41,375 orders, followed by Rio de Janeiro with 12,762 and Minas Gerais with 11,544.`
- **Fixed in B**: Compare revenue in 2017 and 2018.
  - A: `Here is the result (see the table above).`
  - B: `Revenue increased from R$ 6.11 million in 2017 to R$ 7.34 million in 2018.`
- **Fixed in B**: Which 5 cities have the most customers?
  - A: `Here is the result (see the table above).`
  - B: `São Paulo leads with 14,865 customers, followed by Rio de Janeiro with 6,576. Belo Horizonte, Brasília, and Curitiba round out the top five with 2,654, 2,056, a…`
- **Fixed in B**: What is the average review score for each of the top 3 categories by orders?
  - A: `The average review score for "bed_bath_table" is 3.90 R$, the average review score for "health_beauty" is 4.14 R$ , and the average review score for "sports_lei…`
  - B: `The top 3 categories by order volume are bed_bath_table with 9,417 orders and an average review of 3.90, health_beauty with 8,836 orders and an average review o…`
- **Fixed in B**: Which 3 states have the slowest average delivery?
  - A: `The states with the slowest average delivery times are Alagoas (R$28.2 thousand), Rio Grande do Norte (R$28.2 thousand) and Amazonas (R$26.3 thousand).`
  - B: `AP and RR are tied for the slowest average delivery at 28.2 days, while AM follows with 26.3 days.`
- **Broken in B**: How many orders were there in each order status?
  - A: `There were 96,478 orders delivered, 1,106 orders shipped, 461 canceled orders, 312 invoices issued, 301 orders in processing, 6 unavailable orders, and 2 approv…`
  - B: `The most common status is delivered with 96,478 orders, followed by shipped with 1,106 orders. Canceled orders account for 461, while invoiced and processing ea…`
- **Failing in both**: Which 3 states have the fastest average delivery?
  - A: `Here is the result (see the table above).`
  - B: `The fastest average delivery times are in São Paulo (R$ 8.7), Paraná (R$ 11.9), and Minas Gerais (R$ 11.9).`
- **Failing in both**: What were the 3 best months for revenue?
  - A: `Here is the result (see the table above).`
  - B: `Here is the result (see the table above).`
- **Failing in both**: Which 3 categories have the lowest average review score (with at least 100 orders)?
  - A: `The categories with the lowest average review scores are office_furniture at 3.49 R$, fashion_male_clothing at 3.64 R$, and fixed_telephony at 3.68 R$.`
  - B: `office_furniture, fashion_male_clothing, and fixed_telephony have the lowest average review scores at R$3.49, R$3.64, and R$3.68 respectively.`
- **Failing in both**: How many orders were placed in 2020?
  - A: `I don't have enough data to answer that: nothing matched. The data covers Sep 2016 to Oct 2018; check the spelling of any names, or open **SQL used** below to s…`
  - B: `I don't have enough data to answer that: nothing matched. The data covers Sep 2016 to Oct 2018; check the spelling of any names, or open **SQL used** below to s…`

### report_generator
- **Fixed in B**: Report on the whole marketplace
  - A: `* Olist has achieved significant revenue growth year-over-year (138%). * The average order value is relatively high at R$ 137.42. * Late deliveries represent a …`
  - B: `**Key findings** - Revenue is heavily concentrated in São Paulo (38%), with São Paulo and Rio de Janeiro together accounting for over half of total sales. - Hea…`
- **Fixed in B**: Report on watches_gifts
  - A: `* **Key findings**: * Watches_gifts category generated R$ 1.2M in revenue, representing 8.88% of Olist's marketplace sales. * The average order value is R$ 213.…`
  - B: `**Key findings** - Watches & gifts generated R$ 1.20M, representing 8.88% of total marketplace revenue. - Revenue grew 243% year-over-year from Jan-Aug 2017 to …`
- **Fixed in B**: Report on computers_accessories
  - A: `* **Computers accessories revenue represents 6.7% of Olist's marketplace.** * **Revenue grew by +129% year-over-year in Jan-Aug 2018.** * **The category has a h…`
  - B: `**Key findings** - Computers accessories generated R$ 904.3K in revenue, representing 6.7% of total marketplace sales. - Revenue grew 129% from Jan-Aug 2017 to …`
- **Fixed in B**: Report on Rio de Janeiro city
  - A: `* Rio de Janeiro represents 7.31% of Olist's total revenue. * The city has a high customer base with 6,800 orders and an average order value of R$ 145.08. * Lat…`
  - B: `**Key findings** - Rio de Janeiro generates R$ 986.5K in revenue, representing 7.31% of total marketplace sales. - The city shows strong growth momentum, with r…`
- **Fixed in B**: Report on Curitiba
  - A: `* Curitiba generates R$208,000 in revenue, representing 1.54% of Olist's total marketplace sales. * The city has a high customer engagement rate with 1,502 orde…`
  - B: `**Key findings** - Curitiba generated R$ 208.0K in revenue, representing 1.54% of marketplace total with an average order value of R$ 138.49. - The city shows s…`
- **Fixed in B**: Report on Belo Horizonte
  - A: `* Belo Horizonte contributes approximately 2.6% of Olist's total revenue. * The city has seen a significant increase in revenue year-over-year (196%), with peak…`
  - B: `**Key findings** - Belo Horizonte contributes 2.6% of marketplace revenue (R$ 350.8K) with a modest average order value of R$ 128.16. - The city shows strong gr…`
- **Fixed in B**: Report on Sao Paulo state
  - A: `* **SP accounts for 38.27% of Olist's revenue.** * **Average order value is R$ 125.57.** * **Late deliveries in SP account for 5.9% of all delivered orders.** *…`
  - B: `**Key findings** - São Paulo dominates state revenue with R$ 1.90M (37%), while Campinas and Guarulhos contribute 4% and 3% respectively. - Revenue grew 166% fr…`
- **Broken in B**: Report on bed_bath_table
  - A: `* The category generates significant revenue, representing 7.68% of Olist's total marketplace sales. * Revenue growth is strong year-over-year, increasing by 10…`
  - B: `**Key findings** - Bed_bath_table generated R$ 1.04M, representing 7.68% of total marketplace revenue. - Revenue grew 107% from Jan-Aug 2017 to Jan-Aug 2018, in…`
- **Failing in both**: Report on health_beauty
  - A: `* Health & Beauty revenue represents 9.31% of Olist's marketplace sales. * Revenue increased by +211% year-over-year in Jan-Aug 2018 compared to the same period…`
  - B: `**Key findings** - Health & beauty generated R$ 1.26M, representing 9.31% of total marketplace revenue. - Revenue grew 211% from Jan-Aug 2017 to Jan-Aug 2018, r…`
- **Failing in both**: Report on toys
  - A: `* The toys category generated R$479.8K in revenue, representing 3.56% of Olist's marketplace sales. * Orders for toys totaled 3,855 with an average order value …`
  - B: `**Key findings** - Toys generated R$ 479.8K in revenue, representing 3.56% of total marketplace sales. - Revenue grew 42% from Jan-Aug 2017 to Jan-Aug 2018, inc…`
- **Failing in both**: Report on Sao Paulo city
  - A: `* Sao Paulo represents 14% of Olist's total revenue and accounts for 14.76% of customers. * Revenue in Sao Paulo has grown significantly year-over-year by +185%…`
  - B: `**Key findings** - Sao Paulo generated R$ 1.90M in revenue, representing 14.06% of total marketplace sales. - The city shows strong growth, with revenue increas…`
- **Failing in both**: Report on Brasilia
  - A: `* **Brasilia's revenue represents 2.22% of Olist's total.** * **Average order value is R$ 142.21 and average review score is 4.02/5.** * **Late deliveries accou…`
  - B: `**Key findings** - November 2017 drove 161 orders (R$ 26.9K), representing the city's peak activity. - Revenue grew 165% from Jan-Aug 2017 to Jan-Aug 2018 (R$ 6…`
- **Failing in both**: Report on Minas Gerais
  - A: `* Revenue in MG represents 11.66% of Olist's total marketplace revenue. * The average order value is R$ 136.87 with a high customer base (11,134). * Late delive…`
  - B: `**Key findings** - MG contributes 11.66% of total marketplace revenue with 11,496 orders from 11,134 customers. - Revenue is highly concentrated in Belo Horizon…`
- **Failing in both**: Report on Rio Grande do Sul
  - A: `* RS revenue represents 5.5% of Olist's total marketplace revenue. * The average order value in RS is R$ 137.13. * Late deliveries account for 7.1% of orders de…`
  - B: `**Key findings** - RS generates R$ 742.6K, representing 5.5% of total marketplace revenue. - Revenue grew 111% from Jan-Aug 2017 to Jan-Aug 2018, jumping from R…`
- **Failing in both**: Report on Bahia
  - A: `* **BA's revenue is significantly lower than the national average.** It represents just 3.76% of Olist's total marketplace revenue. * **The state has a high num…`
  - B: `**Key findings** - Revenue is heavily concentrated in Salvador, which accounts for 36% of total sales despite representing a small share of orders. - Late deliv…`
