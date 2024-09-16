# Market Basket Analysis with Apriori Algorithm

This project implements market basket analysis using the Apriori algorithm to discover association rules in transaction data.

## Files

1. `my_apriori.py`: Main script for data processing and applying the Apriori algorithm.
2. `apyori.py`: Implementation of the Apriori algorithm.

## Dependencies

- numpy
- matplotlib
- pandas
- apyori

## Usage

1. Ensure you have a CSV file named `Market_Basket_Optimisation.csv` in the same directory as the scripts.
2. Run `my_apriori.py` to perform the analysis.

## How it works

1. The script reads transaction data from `Market_Basket_Optimisation.csv`.
2. Transactions are processed and converted into a suitable format.
3. The Apriori algorithm is applied with the following parameters:
   - Minimum support: 0.003
   - Minimum confidence: 0.2
   - Minimum lift: 3
   - Minimum length: 2

4. Results are stored in the `results` variable as a list of association rules.

## Customization

You can adjust the Apriori algorithm parameters in `my_apriori.py` to fine-tune the analysis:
