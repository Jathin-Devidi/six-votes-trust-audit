# Six Votes audit report

1500 questions, 6 models. All numbers come from `results.json`.

## 1. Is the top model really #1?

| Rank | Model | Accuracy | 95% interval |
|---|---|---|---|
| 1 | atlas-large | 71.4% | 68.9% to 73.7% |
| 2 | nimbus-2 | 69.7% | 67.5% to 72.1% |
| 3 | atlas-medium | 68.9% | 66.6% to 71.3% |
| 4 | borealis-xl | 66.9% | 64.5% to 69.3% |
| 5 | cirrus-7b | 64.9% | 62.6% to 67.4% |
| 6 | borealis-mini | 61.4% | 58.9% to 63.8% |

**atlas-large vs nimbus-2:** gap 1.7 points, 95% paired interval -0.7 to 3.9, exact McNemar p = 0.182. That is NOT a statistically reliable lead.

If the benchmark were drawn again, how often would the same model still come first?

| Questions | Leader holds #1 |
|---|---|
| 50 | 46.7% |
| 200 | 63.2% |
| 500 | 77.2% |
| 1500 | 91.2% |

## 2. How many independent opinions are the models worth?

- Mean error correlation (phi): **0.31**
- Effective number of models: **2.35 of 6**
- Plurality vote accuracy: **77.5%** (best single model 71.4%)
- A majority is right on 64.5% of questions; independent models with the same accuracies would manage 69.1%.
- The average hides structure: **atlas-medium and nimbus-2** have phi 0.87; when both are wrong (418 questions) they give the same wrong answer 95.2% of the time. Two votes from them are close to one vote counted twice.

## 3. Best three-model panel (chosen on dev, scored once on test)

- Chosen panel: atlas-large, borealis-xl, cirrus-7b - dev 76.9%, test 75.3%
- Top-3 by accuracy: atlas-large, nimbus-2, atlas-medium - dev 70.0%, test 70.8%
- The dev score of the chosen panel is optimistic because it was the best of many subsets on that same data; the test score is the honest one.

## 4. Where each model is weakest

| Model | Weakest category | There | Elsewhere |
|---|---|---|---|
| atlas-large | sql (n=311) | 59.2% | 74.6% |
| cirrus-7b | statistics (n=272) | 61.4% | 65.7% |
| borealis-xl | python (n=307) | 64.8% | 67.5% |
| nimbus-2 | ml-concepts (n=304) | 67.8% | 70.2% |
| atlas-medium | statistics (n=272) | 67.3% | 69.2% |
| borealis-mini | statistics (n=272) | 60.3% | 61.6% |

## Caveats

- Question-level bootstrap assumes questions are independent draws from the same population.
- Models that share a vendor or training data can share mistakes; correlation here is measured, not explained.
- Accuracy is measured against the supplied gold answers; wrong gold labels would hit every model alike.
