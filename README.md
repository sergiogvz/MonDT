# MonDT - Decision Trees for Classification with Monotonicity Constraints

This repository implements two different decision trees for monotonic classification:

- Rank Discrimination Measure Tree  (RDMT). This decision tree can be built with three different rank discrimination measures: Gini, Shannon, or Pessimistic.

  Marsala, C., & Petturiti, D. (2015). Rank discrimination measures for enforcing monotonicity in decision tree induction. Information Sciences, 291, 143-171.

- Rank Entropy-Based Decision Trees for Monotonic Classification (REMT)

  Hu, Q., Che, X., Zhang, L., Zhang, D., Guo, M., & Yu, D. (2011). Rank entropy-based decision trees for monotonic classification. IEEE Transactions on Knowledge and Data Engineering, 24(11), 2052-2064.

## Example

The file *exampleMonDT* is an example of the execution of these decision trees with the the Artiset data-set. The results obtained are the followings:

| Algorithm | Accuracy | MAE | NMI |
| -- | -- | -- | -- |
| RDMT_Gini | 0.81 | 0.19 | 0.0 |
| RDMT_Shannon | 0.86 | 0.14 | 0.0 |
| RDMT_Pessimistic | 0.88 | 0.12 | 0.0 |
| REMT | 0.86 | 0.14 | 0.0 |
