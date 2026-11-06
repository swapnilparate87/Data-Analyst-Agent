# Evaluation: `gemma2:2b` vs `gemma2:2b`

- **A**: run `baseline_gemma2-2b`, model `gemma2:2b` (ollama), judge `nvidia/nemotron-3-ultra-550b-a55b`
- **B**: run `gemma2-2b_prompt-fix`, model `gemma2:2b` (ollama), judge `nvidia/nemotron-3-ultra-550b-a55b`

## Cases passed

| Component | A | B | Change | Time A | Time B |
|---|---|---|---|---|---|
| scope_guard | 20/20 (100%) | 20/20 (100%) | +0 | 71s | 47s |
| router | 18/18 (100%) | 18/18 (100%) | +0 | 104s | 75s |
| entity_resolver | 17/18 (94%) | 18/18 (100%) | +1 | 6s | 6s |
| retriever | 13/18 (72%) | 17/18 (94%) | +4 | 162s | 101s |
| answer_generator | 7/16 (44%) | 15/16 (94%) | +8 | 1505s | 1911s |
| report_generator | 1/15 (7%) | 0/15 (0%) | -1 | 1981s | 2859s |
| **total** | **76/105 (72%)** | **88/105 (84%)** | **+12** | | |

## Metrics (pass rate / average score)

| Component | Metric | A | B | Score change |
|---|---|---|---|---|
| scope_guard | Exact match | 100% / 1.00 | 100% / 1.00 | +0.00 |
| router | Exact match | 100% / 1.00 | 100% / 1.00 | +0.00 |
| entity_resolver | Exact match | 94% / 0.94 | 100% / 1.00 | +0.06 |
| retriever | Result match | 72% / 0.78 | 94% / 0.94 | +0.17 |
| answer_generator | Correctness [GEval] | 56% / 0.56 | 94% / 0.94 | +0.38 |
| answer_generator | Faithfulness | 81% / 0.85 | 100% / 1.00 | +0.15 |
| answer_generator | Answer Relevancy | 94% / 0.91 | 100% / 0.98 | +0.07 |
| report_generator | Numbers grounded | 93% / 0.99 | 93% / 0.99 | +0.00 |
| report_generator | Faithfulness | 93% / 0.91 | 100% / 0.98 | +0.07 |
| report_generator | Report quality [GEval] | 7% / 0.37 | 0% / 0.43 | +0.06 |

## Cases that changed

### entity_resolver
- **Fixed in B**: [detect] Compare SP sales with the rest of Brazil
  - A: `city: sales`
  - B: `state: SP`

### retriever
- **Fixed in B**: What was the total revenue in 2017?
  - A: `[{"sum(price)":66650.37},{"sum(price)":27817.58},{"sum(price)":47314.71},{"sum(price)":50912.71},{"sum(price)":5962902.01},{"sum(price)":209.6}]`
  - B: `[{"revenue":6155806.98}]`
- **Fixed in B**: What is the average freight value per item?
  - A: `[{"average_freight_per_item":22.25}]`
  - B: `[{"avg_freight":19.99}]`
- **Fixed in B**: Which 10 sellers made the most revenue in 2018?
  - A: `[{"seller_id":"4869f7a5dfa277a7dca6462dcf3b52b2","total_revenue":138414.6},{"seller_id":"955fee9216a65b617aa5c0531780ce60","total_revenue":117340.86},{"seller_i…`
  - B: `[{"seller_id":"4869f7a5dfa277a7dca6462dcf3b52b2","revenue":138414.6},{"seller_id":"955fee9216a65b617aa5c0531780ce60","revenue":117328.86},{"seller_id":"7d13fca1…`
- **Fixed in B**: What are the 3 states with the highest share of late deliveries?
  - A: `query failed: SELECT state, SUM(CASE WHEN delivered_late = true THEN 1 ELSE 0 END) as late_deliveries_count FROM sales GROUP BY state ORDER BY late_deliveries_c…`
  - B: `[{"state":"AL","late_delivery_pct":23.9},{"state":"MA","late_delivery_pct":19.7},{"state":"PI","late_delivery_pct":16.0}]`
- **Failing in both**: What is the total freight paid by customers in Belo Horizonte?
  - A: `[{"count(DISTINCT sf.freight_value)":1095}]`
  - B: `[{"count(DISTINCT sf.order_id)":2750}]`

### answer_generator
- **Fixed in B**: Which 3 states have the most orders?
  - A: `Here is the result (see the table above).`
  - B: `1. **SP**: orders 41,375 2. **RJ**: orders 12,762 3. **MG**: orders 11,544`
- **Fixed in B**: Compare revenue in 2017 and 2018.
  - A: `Here is the result (see the table above).`
  - B: `1. **2,017.00**: revenue 6,108,492.27 2. **2,018.00**: revenue 7,341,182.41`
- **Fixed in B**: Which 5 cities have the most customers?
  - A: `Here is the result (see the table above).`
  - B: `Sao Paulo has the most customers with 14865 unique users. Rio de Janeiro has 6576 unique users, followed by Belo Horizonte with 2654 unique users. Brasilia and …`
- **Fixed in B**: What is the average review score for each of the top 3 categories by orders?
  - A: `The average review score for "bed_bath_table" is 3.90 R$, the average review score for "health_beauty" is 4.14 R$ , and the average review score for "sports_lei…`
  - B: `1. **bed_bath_table**: orders 9,417, avg_review 3.90 2. **health_beauty**: orders 8,836, avg_review 4.14 3. **sports_leisure**: orders 7,720, avg_review 4.11`
- **Fixed in B**: Which 3 states have the slowest average delivery?
  - A: `The states with the slowest average delivery times are Alagoas (R$28.2 thousand), Rio Grande do Norte (R$28.2 thousand) and Amazonas (R$26.3 thousand).`
  - B: `The three states with the slowest average delivery times are Alagoas (AP), Rio Grande do Sul (RR), and Amazonas (AM). Their average delivery days is 28.2, 28.2,…`
- **Fixed in B**: Which 3 states have the fastest average delivery?
  - A: `Here is the result (see the table above).`
  - B: `1. **SP**: avg_delivery_days 8.70 2. **MG**: avg_delivery_days 11.90 3. **PR**: avg_delivery_days 11.90`
- **Fixed in B**: What were the 3 best months for revenue?
  - A: `Here is the result (see the table above).`
  - B: `1. **2017-11-01**: revenue 1,003,862.14 2. **2018-04-01**: revenue 993,592.98 3. **2018-05-01**: revenue 992,871.75`
- **Fixed in B**: Which 3 categories have the lowest average review score (with at least 100 orders)?
  - A: `The categories with the lowest average review scores are office_furniture at 3.49 R$, fashion_male_clothing at 3.64 R$, and fixed_telephony at 3.68 R$.`
  - B: `1. **office_furniture**: avg_review 3.49 2. **fashion_male_clothing**: avg_review 3.64 3. **fixed_telephony**: avg_review 3.68`
- **Failing in both**: How many orders were placed in 2020?
  - A: `I don't have enough data to answer that: nothing matched. The data covers Sep 2016 to Oct 2018; check the spelling of any names, or open **SQL used** below to s…`
  - B: `I don't have enough data to answer that: nothing matched. The data covers Sep 2016 to Oct 2018; check the spelling of any names, or open **SQL used** below to s…`

### report_generator
- **Broken in B**: Report on bed_bath_table
  - A: `* The category generates significant revenue, representing 7.68% of Olist's total marketplace sales. * Revenue growth is strong year-over-year, increasing by 10…`
  - B: `* The category generates significant revenue, representing 7.68% of Olist's total marketplace sales. * Revenue growth is strong year-over-year, increasing by 10…`
- **Failing in both**: Report on the whole marketplace
  - A: `* Olist has achieved significant revenue growth year-over-year (138%). * The average order value is relatively high at R$ 137.42. * Late deliveries represent a …`
  - B: `* Olist has achieved significant revenue growth year-over-year (138%). * The average order value is relatively high at R$ 137.42. * Late deliveries represent a …`
- **Failing in both**: Report on health_beauty
  - A: `* Health & Beauty revenue represents 9.31% of Olist's marketplace sales. * Revenue increased by +211% year-over-year in Jan-Aug 2018 compared to the same period…`
  - B: `* Health & Beauty revenue represents 9.31% of Olist's marketplace sales. * Revenue increased by +211% year-over-year in Jan-Aug 2018 compared to the same period…`
- **Failing in both**: Report on watches_gifts
  - A: `* **Key findings**: * Watches_gifts category generated R$ 1.2M in revenue, representing 8.88% of Olist's marketplace sales. * The average order value is R$ 213.…`
  - B: `* **Key findings**: * Watches_gifts category generated R$ 1.2M in revenue, representing 8.88% of Olist's marketplace sales. * The average order value is R$ 213.…`
- **Failing in both**: Report on computers_accessories
  - A: `* **Computers accessories revenue represents 6.7% of Olist's marketplace.** * **Revenue grew by +129% year-over-year in Jan-Aug 2018.** * **The category has a h…`
  - B: `* **Computers accessories revenue represents 6.7% of Olist's marketplace.** * **Revenue grew by +129% year-over-year in Jan-Aug 2018.** * **The category has a h…`
- **Failing in both**: Report on toys
  - A: `* The toys category generated R$479.8K in revenue, representing 3.56% of Olist's marketplace sales. * Orders for toys totaled 3,855 with an average order value …`
  - B: `* The toys category generated R$479.8K in revenue, representing 3.56% of Olist's marketplace sales. * Orders for toys totaled 3,855 with an average order value …`
- **Failing in both**: Report on Sao Paulo city
  - A: `* Sao Paulo represents 14% of Olist's total revenue and accounts for 14.76% of customers. * Revenue in Sao Paulo has grown significantly year-over-year by +185%…`
  - B: `* Sao Paulo represents 14% of Olist's total revenue and accounts for 14.76% of customers. * Revenue in Sao Paulo has grown significantly year-over-year by +185%…`
- **Failing in both**: Report on Rio de Janeiro city
  - A: `* Rio de Janeiro represents 7.31% of Olist's total revenue. * The city has a high customer base with 6,800 orders and an average order value of R$ 145.08. * Lat…`
  - B: `* Rio de Janeiro represents 7.31% of Olist's total revenue. * The city has a high customer base with 6,800 orders and an average order value of R$ 145.08. * Lat…`
- **Failing in both**: Report on Curitiba
  - A: `* Curitiba generates R$208,000 in revenue, representing 1.54% of Olist's total marketplace sales. * The city has a high customer engagement rate with 1,502 orde…`
  - B: `* Curitiba generates R$208,000 in revenue, representing 1.54% of Olist's total marketplace sales. * The city has a high customer engagement rate with 1,502 orde…`
- **Failing in both**: Report on Belo Horizonte
  - A: `* Belo Horizonte contributes approximately 2.6% of Olist's total revenue. * The city has seen a significant increase in revenue year-over-year (196%), with peak…`
  - B: `* Belo Horizonte contributes approximately 2.6% of Olist's total revenue. * The city has seen a significant increase in revenue year-over-year (196%), with peak…`
- **Failing in both**: Report on Brasilia
  - A: `* **Brasilia's revenue represents 2.22% of Olist's total.** * **Average order value is R$ 142.21 and average review score is 4.02/5.** * **Late deliveries accou…`
  - B: `* **Brasilia's revenue represents 2.22% of Olist's total.** * **Average order value is R$ 142.21 and average review score is 4.02/5.** * **Late deliveries accou…`
- **Failing in both**: Report on Sao Paulo state
  - A: `* **SP accounts for 38.27% of Olist's revenue.** * **Average order value is R$ 125.57.** * **Late deliveries in SP account for 5.9% of all delivered orders.** *…`
  - B: `* **SP accounts for 38.27% of Olist's revenue.** * **Average order value is R$ 125.57.** * **Late deliveries in SP account for 5.9% of all delivered orders.** *…`
- **Failing in both**: Report on Minas Gerais
  - A: `* Revenue in MG represents 11.66% of Olist's total marketplace revenue. * The average order value is R$ 136.87 with a high customer base (11,134). * Late delive…`
  - B: `* Revenue in MG represents 11.66% of Olist's total marketplace revenue. * The average order value is R$ 136.87 with a high customer base (11,134). * Late delive…`
- **Failing in both**: Report on Rio Grande do Sul
  - A: `* RS revenue represents 5.5% of Olist's total marketplace revenue. * The average order value in RS is R$ 137.13. * Late deliveries account for 7.1% of orders de…`
  - B: `* RS revenue represents 5.5% of Olist's total marketplace revenue. * The average order value in RS is R$ 137.13. * Late deliveries account for 7.1% of orders de…`
- **Failing in both**: Report on Bahia
  - A: `* **BA's revenue is significantly lower than the national average.** It represents just 3.76% of Olist's total marketplace revenue. * **The state has a high num…`
  - B: `* **BA's revenue is significantly lower than the national average.** It represents just 3.76% of Olist's total marketplace revenue. * **The state has a high num…`
