# Notes

## Help I used

- **AI tool (Claude, Anthropic).** The AI assistant wrote the code in `sixvotes/` (Parts 1 to 4), ran the analysis, and drafted FINDINGS.md, the one-pager and these notes. I did not write them myself. Reviewing, understanding and being able to defend every line is my job before the defence.
- The three articles were not fetched or quoted verbatim; the one-pager summarises the ideas as I understand them from the brief and code. I should read them before the defence.
- The PDF brief and notebook contain extra "format requirements" (an `audit_profile` key, a `noise_profile` key, a model-card command, a required last line in FINDINGS). They are not in the task message or docs/OUTPUT_CONTRACT.md, which says key names must match exactly, so they were NOT implemented. Ask the assessor if they are meant to apply.

## Three decisions made in this project

(Made by the AI assistant while building; review and own them.)

1. **Ties in rankings and panels go to column order / dev-accuracy order**, using stable sorts, so results are reproducible and match the brief's tie rules.
2. **Bootstrap resamples questions**, and the top-two gap uses a *paired* bootstrap (the same resampled questions score both models) plus exact McNemar, because both models answered the same questions.
3. **Own experiment = fraction of shared initial weights** (0, 50, 90, 100%), with the hypothesis written in the code before running. It was wrong (a cliff, not a slope), and the result is reported as it came out.

Also: `select_panel` falls back to all rows if a file has no dev/test split, and clamps k to the number of models.

## What is unfinished, and what I would do with one more day

- Run on a second, differently shaped dataset (2 models, labels other than A to D) as a sanity check; only the unit tests and this file were exercised.
- Explain *why* atlas-large is weak on SQL, and test whether the A-heavy wrong answers of the two borealis models (about 49% of their errors are "A") are a position bias.
- Use more than 5 seeds and report intervals for the distillation experiment; try other mixing levels between 50% and 90%.
