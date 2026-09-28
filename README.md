# Factory-to-Customer Shipping Route Efficiency Analysis

## Project
Nassau Candy Distributor — Factory-to-Customer Shipping Route Efficiency Analysis

## Objective
Analyze historical shipping/order data to understand factory-to-customer route patterns, shipment volumes, shipping-mode patterns, costs, and route-level business metrics. A Streamlit dashboard provides interactive exploration and a Random Forest regression model provides a shipment-cost prediction component.

## Dataset
The project uses the Nassau Candy Distributor CSV dataset containing 10,194 shipment/order records and 19 original columns.

Original fields include:
- Row ID
- Order ID
- Order Date
- Ship Date
- Ship Mode
- Customer ID
- Country/Region
- City
- State/Province
- Postal Code
- Division
- Region
- Product ID
- Product Name
- Sales
- Units
- Gross Profit
- Cost

## Project Structure
```text
nassau-candy-project/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── shipping_data.csv
│   ├── processed_shipping_data.csv
│   ├── route_performance.csv
│   └── random_forest_cost_model.pkl
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   └── feature_engineering.py
├── requirements.txt
└── README.md
```

## How to Run
Open PowerShell or Command Prompt:

```powershell
cd D:\nassau-candy-project
streamlit run app\streamlit_app.py
```

The dashboard should open at:

```text
http://localhost:8501
```

## Main Dashboard Sections
1. Overview
2. Route Analysis
3. Shipping Analysis
4. Cost Prediction
5. ML Model Performance

## Main Results
- Total shipments: 10,194
- Total sales: 141,783.63
- Total cost: 48,340.83
- Gross profit: 93,442.80
- Average calculated lead time: 1320.84 days
- Standard Class shipments: 6,120
- First Class shipments: 1,548
- Second Class shipments: 1,979
- Same Day shipments: 547

Factory shipment counts:
- Lot's O' Nuts: 5,692
- Wicked Choccy's: 4,152
- Secret Factory: 217
- The Other Factory: 100
- Sugar Shack: 33

## Machine Learning
The project includes a Random Forest regression model for shipment Cost.

Features used:
- Ship Mode
- Factory
- State/Province
- Region
- Division
- Product Name
- Units
- Sales

Random Forest test results:
- MAE: 0.00636
- RMSE: 0.08487
- R²: 0.99962

Five-fold cross-validation:
- Mean R²: 0.99249
- Standard deviation: 0.00926

A secondary model without Sales was also evaluated and achieved R² = 0.99950 on the held-out test set.

## Important Model Interpretation
The very high Random Forest performance is specific to this dataset and should not be interpreted as proof of equivalent real-world performance. Sales and Units have strong relationships with Cost in this dataset.

Ship Mode has very low feature importance in the trained cost model. Testing the same shipment under all four shipping modes produced the same predicted cost of $2.28 for the selected test row. Therefore, the dashboard should use historical shipping-mode analysis separately from the ML prediction rather than forcing the model to produce different costs for different modes.

## Data Quality Note
The source dates produce unusually large calculated differences between Order Date and Ship Date. The project preserves the source dates and reports the calculated Lead Time as derived from those dates. These values should therefore be interpreted as dataset-derived values rather than assumed to represent normal real-world delivery durations.

## Factory Mapping
Product names are mapped to factories using the project specification and the source dataset. Factory coordinates are included for the specified factories. Geographic distance should be treated as approximate if destination coordinates are estimated from state/region information.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib / Seaborn
- Streamlit
- Jupyter Notebook

## Disclaimer
This is an analytical prototype based on the supplied historical dataset. Findings describe patterns in that dataset and should not be treated as causal conclusions or guaranteed future logistics outcomes.
