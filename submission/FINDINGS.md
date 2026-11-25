# Findings

## Recommendation

**Yes, with changes.** Ship the vote, but not as "six independent graders", and do not pay for all six.

- The vote scores **77.5%** against **71.4%** for the best single model (atlas-large). The 6.1-point gain is solid (95% interval 4.3 to 8.0).
- The six models are worth only **2.35 independent opinions** (mean error correlation 0.31). atlas-medium and nimbus-2 are near-duplicates: errors correlate 0.87, and when both are wrong they give the same wrong answer 95% of the time.
- A panel of three (atlas-large, borealis-xl, cirrus-7b), chosen on dev and scored once on test, gets **75.3%**. The six-model vote gets 77.6% on the same test rows; that difference is not reliable (p = 0.08). Three models give most of the benefit.

Changes: drop one of the duplicate pair, stop calling atlas-large "the best" (its lead over nimbus-2 is 1.7 points, p = 0.18, and it stays #1 on a redrawn benchmark only 91% of the time), and keep a human check on low-agreement answers.

## One thing you did not ask about

The top model has a hole. atlas-large scores **74.6%** on everything except SQL, but only **59.2%** on SQL (311 questions). On SQL it is below four of the five other models. An overall leaderboard hides this, and a team grading SQL candidates would be using the weakest voter at its weakest. The vote itself is also lower on SQL (72.0%) than anywhere else. Check each category, not only the total.

## My experiment

**Question.** How much of the teacher's starting weights must a noise-trained student share for the hidden skill to transfer?

**Expected.** A smooth fade: 50% shared gives about half the gain.

**What happened.** Averaged over five seeds, student accuracy on real digits was 11% at 0% shared, **9% at 50%**, 70% at 90%, and 92% at 100% shared. It is a cliff, not a slope.

**What it means.** If you buy models from several vendors, hidden traits pass between models only when they start from nearly identical weights, i.e. the same base model. Mixing vendors is a real protection; two fine-tunes of one base are not independent.
