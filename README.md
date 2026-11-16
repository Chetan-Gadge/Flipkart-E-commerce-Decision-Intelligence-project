Domain: E-commerce

# Flipkart E-commerce Decision Intelligence

End-to-end customer analytics pipeline for order-value estimation, churn prediction, customer segmentation, and downstream neural-network sizing, built in Python and NumPy.

## How to run

Run the following commands in order from the repository root on a clean clone:

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

## Part 1 findings

- **R1.1 (Hypothesis test & practical significance):** H0 (μ = 8.0, or INR 800) was **rejected** because `z_stat = 2.8930` exceeds the two-tailed 5% critical value of `1.96`. In this domain, a Type I error means falsely concluding that Flipkart's true average order value differs from INR 800 when it actually equals INR 800, prompting unnecessary category-plan adjustments. The observed gap is `mean_value - 8.0 = 0.9382` column units (INR 93.82), which is practically meaningful as well—nearly 12% above the INR 800 planning assumption.
- **R1.2 (Sample size):** `n_required = 1293` compared against `800` available rows. Because 800 is below 1,293, the current sample cannot achieve a ±0.50 column-unit (INR 50) margin of error at 95% confidence (its actual half-width is ±0.6356, or INR 63.56), so the planning team needs 493 additional records to meet its precision target.
- **R1.3 (Gradient descent convergence):** `gd_slope_gap = 0.0000`. Gradient descent reached the closed-form line because ordinary least squares is a strictly convex quadratic objective with a unique global minimum, and standardising `X4` keeps the curvature well-conditioned for 500 updates at η = 0.1.
- **R1.4 (Funnel check & log remedy):** `resid_sd_ratio_raw = 3.7525` and `resid_sd_ratio_log = 1.0396`. The raw model exhibits a pronounced funnel—residual spread in the upper half is 3.75 times that in the lower half—and the log-log transformation removes this heteroscedasticity, bringing the ratio down to `1.0396`.
- **R1.5 (Multiple regression predictors):** Applying the mechanical `|beta| > 0.05` rule (excluding the constant) selects `ln(X4)` (`mr_beta_1 = 1.1576`), `X2` (`mr_beta_3 = 0.0940`), `X3` (`mr_beta_4 = 0.1067`), and `X6` (`mr_beta_6 = 0.1518`). Comparing `mr_adj_r2 = 0.7403` with `loglog_adj_r2 = 0.6561` shows a +0.0842 gain in adjusted R², confirming that the extra predictors earn their place.

## Part 2 findings

- **R2.1 (Learning-rate effect):** Ordered from lowest to highest training loss after 200 updates: `loss_epoch_200_eta_1.0 = 0.3703` < `loss_epoch_200_eta_0.1 = 0.3776` < `loss_epoch_200_eta_0.01 = 0.5091`. Too small a learning rate (`0.01`) converges too slowly within 200 steps, while a larger rate (`1.0`) makes rapid progress on this standardised convex loss without diverging.
- **R2.2 (Key predictors & odds ratios):** The three predictors with the largest absolute weight are `X3` (`discount_order_share`, `0.9013`), `X1` (`customer_tenure_months`, `-0.5554`), and `X2` (`complaint_count_90d`, `0.5505`). The largest odds ratio is `odds_ratio_X3 = 2.4629`, meaning each one-SD increase in a customer's discount order share multiplies their odds of churning by 2.4629, holding all other predictors fixed.
- **R2.3 (Accuracy vs. recall/precision):** `blind_accuracy = 0.7875` compared to `tau_0.5_accuracy = 0.8042`. Accuracy alone is a weak score because only 51 of 240 test customers churned, so predicting zero churn for everyone already scores 78.75%. Lowering τ from `0.5` to `0.3` increased recall from `0.4706` to `0.7451` while decreasing precision from `0.5455` to `0.4634`.
- **R2.4 (Logistic vs. KNN):** On the test set, logistic regression at τ = 0.3 achieves `tau_0.3_f1 = 0.5714` and `tau_0.3_recall = 0.7451`, clearly outperforming KNN (`knn_best_k = 5`), which achieves `knn_f1 = 0.3696` and `knn_recall = 0.3333`.
- **R2.5 (Deployment choice & cost):** Deploy **logistic regression** at the cost-minimising threshold `cost_tau = 0.1500`. On the test set, this yields `cost_test_at_tau = INR 6050`, cutting retention cost nearly in half compared to `cost_test_at_0.5 = INR 11800` and saving INR 14,350 relative to `cost_test_flag_nobody = INR 20400`. The threshold was selected on the training rows so the test set remains an unbiased holdout evaluation without threshold-selection leakage.

## Part 3 findings

- **R3.1 (K-means elbow):** The four successive WCSS drops are `WCSS(2) - WCSS(3) = 659.8282`, `WCSS(3) - WCSS(4) = 108.0739`, `WCSS(4) - WCSS(5) = 171.0575`, and `WCSS(5) - WCSS(6) = 45.5466`. The elbow rule selects `elbow_k = 3`, where the drop falls sharply from 659.8282 to 108.0739 (below half); the non-monotonic rise at K = 4→5 occurs because the pinned starting rows yield one reproducible solution rather than necessarily the global optimum for every K.
- **R3.2 (Segment profiles):**
  - **Segment A — Premium Loyal (`size = 322`):** Longest tenure (`47.5028` months), lowest complaints (`1.9503`), lowest discount share (`0.2250`), highest loyalty (`62.1584`), `outcome_rate = 0.0186`, `mean_value = 3.5976` (INR 359.76). *Action:* Maintain frictionless VIP service and avoid unnecessary discount coupons.
  - **Segment B — High-Value At-Risk (`size = 255`):** Mid tenure (`28.6071` months), highest complaints (`6.0706`), highest order values (`mean_X4 = 28.7192`, `mean_value = 17.1248` / INR 1,712.48), `outcome_rate = 0.2353`. *Action:* Prioritise proactive support recovery to resolve complaints and protect high-margin baskets.
  - **Segment C — Bargain-Focused (`size = 223`):** Shortest tenure (`14.3892` months), highest discount share (`0.7118`), lowest loyalty (`38.5215`), `mean_value = 7.2883` (INR 728.83), and the **highest churn rate** at `outcome_rate = 0.4664`. *Action:* Target with early retention coupons and onboarding nudges before the 90-day window closes.
- **R3.3 (Hierarchical clustering cut):** Single linkage (`single_clusters_at_cut = 4`) produces `{E2, E3, E6}`, `{E1}`, `{E4}`, and `{E5}` (cut in the `2.8 - 1.4 = 1.4` gap after merge 2). Complete linkage (`complete_clusters_at_cut = 2`) produces `{E1, E4, E5}` and `{E2, E3, E6}` (cut in the `7.9 - 2.9 = 5.0` gap after merge 4). They disagree because single linkage merges `{E1, E4, E5}` with `{E2, E3, E6}` at their closest pair (`3.4`), leaving a small final gap (`0.5`), whereas complete linkage merges them at their farthest pair (`7.9`), creating a large `5.0` gap at the top.
- **R3.4 (PCA):** After standardising, the mean vector is `[0, 0, 0, 0, 0]`. PC1 loads positively on tenure (`X1 = 0.5317`) and loyalty (`X5 = 0.4418`) and negatively on discount share (`X3 = -0.5187`), complaints (`X2 = -0.4100`), and past order value (`X4 = -0.2916`), capturing a loyal-vs-discount/friction axis. `components_for_80pct = 3` (`cumulative_ratio_3 = 0.8616`). Segment PC1 means (`mean_pc1_segment_A = 1.7773`, `mean_pc1_segment_B = -0.7831`, `mean_pc1_segment_C = -1.6709`) show PC1 cleanly separates Premium Loyal shoppers from High-Value At-Risk and Bargain-Focused shoppers.

## Part 4 findings

- **R4.1 (Purest split & tree overfitting):** `S4` (`X1 <= 24`) is the best split by both information gain (`0.1252`) and Gini gain (`0.0606`) because it isolates 86 of 119 training churners into 188 rows (`45.74%` positive vs `8.87%` for `X1 > 24`). Growing a tree until every leaf is pure overfits by memorising training noise and rare outliers that do not generalise. Setting a maximum depth (`max_depth`) or minimum leaf size (`min_samples_leaf`) prevents excessive depth.
- **R4.2 (Structured models & voting):** Ship **logistic** (`f1 = 0.5714`, `recall = 0.7451`). Combining models did not help: `hard_vote` (`f1 = 0.5344`, `recall = 0.6863`) and `soft_vote` (`f1 = 0.5507`, `recall = 0.7451`) both underperformed standalone logistic regression because KNN (`f1 = 0.3696`, `recall = 0.3333`) and the stump (`f1 = 0.5532`, `recall = 0.7647`) diluted ensemble precision or vote agreement.
- **R4.3 (AdaBoost):** After round 1 normalisation, the two misclassified points (points 3 and 7) carry total weight `0.2500 + 0.2500 = 0.5000`. Round 2 trusts `h2` (`round2_alpha = 0.8047`) more than `h1` (`round1_alpha = 0.5493`) because `h2` achieves a lower weighted error (`0.1667` vs `0.2500`) on the reweighted points.
- **R4.4 (Neural-network sizing & theory):**

| Layer (Model A, Option 2) | Output shape (H × W × D) | Parameters |
|---|---|---:|
| Input | 28 × 28 × 1 | 0 |
| `cnn_conv1` (5×5, 6 filters, s=1, p=2) + ReLU | 28 × 28 × 6 | 156 |
| MaxPool 1 (2×2, s=2) | 14 × 14 × 6 | 0 |
| `cnn_conv2` (5×5, 16 filters, s=1, p=0) + ReLU | 10 × 10 × 16 | 2416 |
| MaxPool 2 (2×2, s=2) | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| `cnn_dense1` (32 neurons, ReLU) | 32 | 12832 |
| `cnn_dense2` (10 neurons) | 10 | 330 |

For Model B, the weight matrix shapes are `W_xh`: `128 × 64`, `W_hh`: `128 × 128`, and `W_hy`: `16 × 128`; `rnn_total = 26768` does not grow with sequence length because the same recurrent weights are shared across every time step. For 10-class classification, use **ReLU** in hidden layers and **Softmax** at the output with **categorical cross-entropy** loss. In very deep plain networks, repeated chain-rule multiplication of small derivatives shrinks gradients exponentially, causing early layers to stop learning; skip connections add the input directly to a block's output (`F(x) + x`), providing an unimpeded gradient path back to early layers.

## Model-choice memo

**To:** Head of Retention Team  
**From:** Customer Analytics  
**Subject:** Production Churn Model Recommendation  

I recommend deploying **logistic regression** from Task 4.2 rather than KNN, the decision stump, or either voting ensemble. On the 240-row test set, logistic regression delivers the highest F1 (`0.5714`) with strong recall (`0.7451`), outperforming KNN (`f1 = 0.3696`), hard voting (`f1 = 0.5344`), and soft voting (`f1 = 0.5507`).

For retention outreach, we should set the alert threshold at **`0.15`** (`cost_tau = 0.1500`), chosen on the training set under the INR 50 false-alarm / INR 400 missed-churn cost rule. On the test set, this threshold lowers total retention cost to **`INR 6050`**, nearly halving the **`INR 11800`** cost of the default `0.5` threshold and saving **`INR 14350`** compared to flagging nobody (`INR 20400`).

The main limitation of this evidence is that evaluation rests on a single 240-row holdout split containing only 51 churn events, and the underlying records are synthetic rather than live Flipkart transactions.

Neither Model A (product-image CNN, 15,734 parameters) nor Model B (review-text RNN, 26,768 parameters) is worth building before the structured churn model is in production, because the structured pipeline already captures substantial retention savings at minimal engineering cost.

## Limitations

1. **Synthetic data:** All 800 rows come from a synthetic generator (`generate_data.py`), so feature distributions and churn relationships do not reflect real Flipkart customer behaviour, seasonality, or catalogue dynamics.
2. **Single holdout split & small positive count:** Test metrics and cost comparisons rest on one 70/30 split with only 240 test rows and 51 positive churn events, making precision, recall, and INR cost estimates sensitive to small sample fluctuations.
3. **Static snapshot & linear log-odds:** The classifier models churn as a linear function of seven static point-in-time features without interaction terms, temporal purchase trends, or post-coupon uplift measurement.
