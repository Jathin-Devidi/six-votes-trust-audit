# One page: what I took from each article

## 1. "Is the #1 AI Model Really #1?" - uncertainty on a leaderboard

**Took:** A single accuracy is a sample. Put a margin of error on it, compare the top two on the *same* questions (paired), and ask how often the ranking would flip on a different set of questions.

**Used in:** Part 1, `leaderboard.py`. Bootstrap intervals per model (atlas-large 71.4%, 68.9 to 73.7). Paired bootstrap and exact McNemar for #1 vs #2: gap 1.7 points, interval -0.7 to 3.9, p = 0.18, so not significant. Rank stability: the leader stays #1 on only 46.7% of 50-question redraws and 91.2% at the full 1,500.

## 2. "Your Six AI Models Are Really Two" - correlated mistakes

**Took:** Votes only help to the extent that errors are independent. Measure error correlation (phi) and shrink the head-count: n_eff = n / (1 + (n-1) x mean phi). An average can hide a near-duplicate pair.

**Used in:** Part 2, `panel.py`. Mean phi 0.31 gives n_eff = 2.35 of 6. atlas-medium and nimbus-2 have phi 0.87 and match on 95% of shared mistakes. A majority is right on 64.5% of questions against 69.1% if the models were independent. Panels are chosen on dev and scored once on test (chosen panel 75.3% test vs 70.8% for the naive top three).

## 3. "A Neural Network Learned to Read Digits From Pure Noise" - hidden channels

**Took:** A student trained only to copy a teacher's meaningless auxiliary outputs on random noise can still learn the teacher's task, but only when it shares the teacher's initial weights. Behaviour can leak through channels that look unrelated to it.

**Used in:** Part 3, `distill.py`. Over 5 seeds the same-start student reached 92.1% on real test digits (untrained 8.1%, teacher 97.0%) while the different-start student stayed at 11.2%. My own experiment varied how much of the start is shared: 9% at 50%, 70% at 90%, 92% at 100%. Transfer is a cliff.

**Link to the project:** if the six models share a base, they may share hidden traits and mistakes, which is what the phi matrix measures from the outside.
