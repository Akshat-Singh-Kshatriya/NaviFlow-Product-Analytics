# Digital Lending Product Performance Engine

**An end-to-end data engineering and product analytics project designed to extract actionable insights for digital lending platforms.**

## Project Overview
This project analyzes a lending product’s user funnel, repayment performance, and risk segments to recommend actionable product feature changes. It demonstrates the ability to handle millions of rows of real-world financial data, clean it rigorously, structure it in a relational database, and use SQL queries to drive product strategy.

The analysis focuses on three core pillars:
1. **Top-of-Funnel Conversion:** Tracking where potential borrowers drop off during the application process.
2. **Risk & Affordability Pricing:** Assessing if the interest rates accurately cover the mathematical risk of default across different user budget segments.
3. **Loan Purpose Optimization:** Identifying which use-cases are loss-leaders and which drive profitable, reliable revenue.

## Data Source
* **Dataset:** [Lending Club Loan Data (2007-2018)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) 
* **Details:** The project utilizes two massive, highly numerical datasets:
  * `accepted_2007_to_2018Q4.csv` (Approved loans with repayment/default status)
  * `rejected_2007_to_2018Q4.csv` (Declined applications with requested amounts and risk profiles)

## Tech Stack
* **Python (Pandas, NumPy):** Handled all out-of-core data ingestion, missing value imputation, outlier treatment, and feature engineering (e.g., calculating *Income-to-Installment Ratios*). 
* **PostgreSQL:** Served as the core analytics engine. Leveraged Common Table Expressions (CTEs), Window Functions (`NTILE`), and complex aggregations to execute business logic and extract insights from the cleaned datasets.

## Key Achievements & Product Insights
By bridging data engineering and product sense, I successfully translated raw loan data into the following strategic product recommendations:

* **Identified the "Thin-File" Funnel Drop-off:** * *Data:* Users with `< 1 year` of employment have an approval rate of just **0.82%**, compared to **64.24%** for those with `10+ years`.
  * *Product Impact:* Recommended implementing an "Alternative Data" UI flow (e.g., account aggregators/bank statement fetch) specifically triggered for users with short employment histories to safely capture this lost top-of-funnel demand.
* **Exposed an Affordability Pricing Gap:** * *Data:* Segmented users into quartiles based on an engineered Income-to-Installment ratio. The tightest budget cohort (Quartile 4) had a default rate of **15.97%**, but was only charged an average interest rate of **14.95%**.
  * *Product Impact:* Proposed a dynamic "Affordability Slider" in the UI that automatically caps requested loan amounts to keep users in a safer repayment tier, rather than issuing underpriced high-risk loans.
* **Segmented High-Risk Loan Purposes:** * *Data:* Discovered that `small_business` loans are massive loss leaders, generating a negative risk-adjusted spread of **-3.58** with an **18.84%** default rate.
  * *Product Impact:* Recommended pausing standard retail flows for business loans and developing a strictly gated, dedicated "B2B / SME" flow that requires different cash-flow verification mechanisms.
