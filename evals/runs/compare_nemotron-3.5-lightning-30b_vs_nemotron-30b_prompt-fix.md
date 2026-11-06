# Evaluation: `nvidia/nemotron-3.5-lightning-30b-a3b` vs `nvidia/nemotron-3.5-lightning-30b-a3b`

- **A**: run `nemotron-3.5-lightning-30b`, model `nvidia/nemotron-3.5-lightning-30b-a3b` (nvidia), judge `nvidia/nemotron-3-ultra-550b-a55b`
- **B**: run `nemotron-30b_prompt-fix`, model `nvidia/nemotron-3.5-lightning-30b-a3b` (nvidia), judge `nvidia/nemotron-3-ultra-550b-a55b`

## Cases passed

| Component | A | B | Change | Time A | Time B |
|---|---|---|---|---|---|
| answer_generator | 11/16 (69%) | 16/16 (100%) | +5 | 1564s | 2076s |
| entity_resolver | 17/18 (94%) | 18/18 (100%) | +1 | 4s | 7s |
| report_generator | 7/15 (47%) | 5/15 (33%) | -2 | 6101s | 1818s |
| retriever | 10/18 (56%) | 18/18 (100%) | +8 | 63s | 89s |
| router | 18/18 (100%) | 18/18 (100%) | +0 | 26s | 29s |
| scope_guard | 20/20 (100%) | 20/20 (100%) | +0 | 19s | 40s |
| **total** | **83/105 (79%)** | **95/105 (90%)** | **+12** | | |

## Metrics (pass rate / average score)

| Component | Metric | A | B | Score change |
|---|---|---|---|---|
| answer_generator | Correctness [GEval] | 81% / 0.85 | 100% / 1.00 | +0.15 |
| answer_generator | Faithfulness | 88% / 0.89 | 100% / 0.99 | +0.09 |
| answer_generator | Answer Relevancy | 100% / 0.99 | 100% / 0.98 | -0.00 |
| entity_resolver | Exact match | 94% / 0.94 | 100% / 1.00 | +0.06 |
| report_generator | Numbers grounded | 80% / 0.98 | 73% / 0.96 | -0.02 |
| report_generator | Faithfulness | 93% / 0.90 | 93% / 0.97 | +0.07 |
| report_generator | Report quality [GEval] | 67% / 0.64 | 53% / 0.63 | -0.01 |
| retriever | Result match | 56% / 0.58 | 100% / 1.00 | +0.42 |
| router | Exact match | 100% / 1.00 | 100% / 1.00 | +0.00 |
| scope_guard | Exact match | 100% / 1.00 | 100% / 1.00 | +0.00 |

## Cases that changed

### answer_generator
- **Fixed in B**: Which 3 states have the fastest average delivery?
  - A: `The fastest average delivery times are in São Paulo (R$ 8.7), Paraná (R$ 11.9), and Minas Gerais (R$ 11.9).`
  - B: `1. **SP**: avg_delivery_days 8.70 2. **MG**: avg_delivery_days 11.90 3. **PR**: avg_delivery_days 11.90`
- **Fixed in B**: How many orders were there in each order status?
  - A: `The most common status is delivered with 96,478 orders, followed by shipped with 1,106 orders. Canceled orders account for 461, while invoiced and processing ea…`
  - B: `1. **delivered**: orders 96,478 2. **shipped**: orders 1,106 3. **canceled**: orders 461 4. **invoiced**: orders 312 5. **processing**: orders 301 6. **unavaila…`
- **Fixed in B**: What were the 3 best months for revenue?
  - A: `Here is the result (see the table above).`
  - B: `1. **2017-11-01**: revenue 1,003,862.14 2. **2018-04-01**: revenue 993,592.98 3. **2018-05-01**: revenue 992,871.75`
- **Fixed in B**: Which 3 categories have the lowest average review score (with at least 100 orders)?
  - A: `office_furniture, fashion_male_clothing, and fixed_telephony have the lowest average review scores at R$3.49, R$3.64, and R$3.68 respectively.`
  - B: `The lowest average review scores belong to office_furniture (3.49), fashion_male_clothing (3.64), and fixed_telephony (3.68).`
- **Fixed in B**: How many orders were placed in 2020?
  - A: `I don't have enough data to answer that: nothing matched. The data covers Sep 2016 to Oct 2018; check the spelling of any names, or open **SQL used** below to s…`
  - B: `I don't have enough data to answer that: nothing matched. The data covers Sep 2016 to Oct 2018; check the spelling of any names, or open **SQL used** below to s…`

### entity_resolver
- **Fixed in B**: [detect] Compare SP sales with the rest of Brazil
  - A: `city: sales`
  - B: `state: SP`

### report_generator
- **Fixed in B**: Report on toys
  - A: `**Key findings** - Toys generated R$ 479.8K in revenue, representing 3.56% of total marketplace sales. - Revenue grew 42% from Jan-Aug 2017 to Jan-Aug 2018, inc…`
  - B: `**Key findings** - Toys generated R$ 479.8K in revenue, representing 3.56% of the marketplace total. - Revenue grew 42% from Jan-Aug 2017 to Jan-Aug 2018, incre…`
- **Fixed in B**: Report on Sao Paulo city
  - A: `**Key findings** - Sao Paulo generated R$ 1.90M in revenue, representing 14.06% of total marketplace sales. - The city shows strong growth, with revenue increas…`
  - B: `**Key findings** - Sao Paulo generates 14.06% of marketplace revenue with an average order value of R$ 124.06. - 6.3% of delivered orders were late, and 9% of r…`
- **Broken in B**: Report on computers_accessories
  - A: `**Key findings** - Computers accessories generated R$ 904.3K in revenue, representing 6.7% of total marketplace sales. - Revenue grew 129% from Jan-Aug 2017 to …`
  - B: `**Key findings** - Computers accessories generated R$ 904.3K, representing 6.7% of total marketplace revenue. - Revenue grew 129% from Jan-Aug 2017 to Jan-Aug 2…`
- **Broken in B**: Report on Curitiba
  - A: `**Key findings** - Curitiba generated R$ 208.0K in revenue, representing 1.54% of marketplace total with an average order value of R$ 138.49. - The city shows s…`
  - B: `**Key findings** - Health & beauty and sports/leisure dominate revenue, together representing 20% of Curitiba's total sales. - Late deliveries affect 5% of orde…`
- **Broken in B**: Report on Belo Horizonte
  - A: `**Key findings** - Belo Horizonte contributes 2.6% of marketplace revenue (R$ 350.8K) with a modest average order value of R$ 128.16. - The city shows strong gr…`
  - B: `**Key findings** - Belo Horizonte contributes 2.6% of marketplace revenue (R$ 350.8K) with 2,737 orders from 2,641 customers. - Health & beauty and bed & bath c…`
- **Broken in B**: Report on Sao Paulo state
  - A: `**Key findings** - São Paulo dominates state revenue with R$ 1.90M (37%), while Campinas and Guarulhos contribute 4% and 3% respectively. - Revenue grew 166% fr…`
  - B: `**Key findings** - São Paulo dominates revenue with R$ 1.90M (37%), while Campinas and Guarulhos contribute R$ 186.9K and R$ 142.6K respectively. - Revenue grew…`
- **Failing in both**: Report on health_beauty
  - A: `**Key findings** - Health & beauty generated R$ 1.26M, representing 9.31% of total marketplace revenue. - Revenue grew 211% from Jan-Aug 2017 to Jan-Aug 2018, r…`
  - B: `**Key findings** - Health & beauty generated R$ 1.26M, representing 9.31% of total marketplace revenue. - Revenue grew 211% year-to-date in 2018 compared to 201…`
- **Failing in both**: Report on bed_bath_table
  - A: `**Key findings** - Bed_bath_table generated R$ 1.04M, representing 7.68% of total marketplace revenue. - Revenue grew 107% from Jan-Aug 2017 to Jan-Aug 2018, in…`
  - B: `**Key findings** - Bed_bath_table generates R$ 1.04M, representing 7.68% of total marketplace revenue. - Revenue grew 107% from Jan-Aug 2017 to Jan-Aug 2018, in…`
- **Failing in both**: Report on Brasilia
  - A: `**Key findings** - November 2017 drove 161 orders (R$ 26.9K), representing the city's peak activity. - Revenue grew 165% from Jan-Aug 2017 to Jan-Aug 2018 (R$ 6…`
  - B: `**Key findings** - November 2017 drove 165 orders, generating R$ 26.9K (8.9% of annual revenue) and marking the city's peak activity. - Health & beauty and watc…`
- **Failing in both**: Report on Minas Gerais
  - A: `**Key findings** - MG contributes 11.66% of total marketplace revenue with 11,496 orders from 11,134 customers. - Revenue is highly concentrated in Belo Horizon…`
  - B: `**Key findings** - MG contributes 11.66% of marketplace revenue with 11,496 orders from 11,134 customers, generating an average order value of R$ 136.87. - Reve…`
- **Failing in both**: Report on Rio Grande do Sul
  - A: `**Key findings** - RS generates R$ 742.6K, representing 5.5% of total marketplace revenue. - Revenue grew 111% from Jan-Aug 2017 to Jan-Aug 2018, jumping from R…`
  - B: `**Key findings** - RS generates R$ 742.6K, representing 5.5% of total marketplace revenue. - Revenue grew 111% from Jan-Aug 2017 to Jan-Aug 2018, jumping from R…`
- **Failing in both**: Report on Bahia
  - A: `**Key findings** - Revenue is heavily concentrated in Salvador, which accounts for 36% of total sales despite representing a small share of orders. - Late deliv…`
  - B: `**Key findings** - Revenue is heavily concentrated in Salvador, which accounts for 36% of total sales despite representing a small share of marketplace volume. …`

### retriever
- **Fixed in B**: How many orders were delivered late?
  - A: `[{"late_deliveries":8715}]`
  - B: `[{"late_orders":7827}]`
- **Fixed in B**: How many orders came from Curitiba?
  - A: `model error: { "sql": "SELECT COUNT(DISTINCT order_id) AS order_count FROM sales WHERE customer_city = 'curitiba' AND order_status NOT IN ('canceled', 'unavaila…`
  - B: `[{"orders":1510}]`
- **Fixed in B**: What is the average review score in furniture_decor?
  - A: `model error: { "sql": "SELECT AVG(review_score) AS avg_review_score FROM sales WHERE category = 'furniture_decor' AND review_score IS NOT NULL; SELECT COUNT(*) …`
  - B: `[{"avg_review_score":3.91}]`
- **Fixed in B**: How many orders were canceled?
  - A: `[{"canceled_orders":244}]`
  - B: `[{"canceled_orders":461}]`
- **Fixed in B**: How many orders came from the state of Rio Grande do Sul?
  - A: `model error: { "sql": "SELECT COUNT(DISTINCT order_id) AS order_count FROM sales WHERE customer_state = 'RS' AND order_status NOT IN ('canceled', 'unavailable')…`
  - B: `[{"orders":5432}]`
- **Fixed in B**: What is the total freight paid by customers in Belo Horizonte?
  - A: `model error: { "sql": "SELECT SUM(freight_value) AS total_freight FROM sales WHERE customer_city = 'belo horizonte' AND order_status NOT IN ('canceled', 'unavai…`
  - B: `[{"total_freight":60710.29}]`
- **Fixed in B**: How many different products are in the health_beauty category?
  - A: `model error: { "sql": "SELECT COUNT(DISTINCT product_id) AS product_count FROM sales WHERE category = 'health_beauty' AND order_status NOT IN ('canceled', 'unav…`
  - B: `[{"products":2444}]`
- **Fixed in B**: What are the 3 states with the highest share of late deliveries?
  - A: `[{"customer_state":"AL","total_deliveries":444,"late_deliveries":103.0,"late_delivery_pct":23.2},{"customer_state":"MA","total_deliveries":824,"late_deliveries"…`
  - B: `[{"state":"AL","late_delivery_pct":23.9},{"state":"MA","late_delivery_pct":19.7},{"state":"PI","late_delivery_pct":16.0}]`
