# Resume Bullets

## Recommended version

**Adaptive DSP Routing Under Capacity Constraints — Contextual Bandits**

- Built a semi-synthetic ad-routing simulator on 100,000 chronological Avazu mobile-ad
  contexts, modeling heterogeneous DSP values, serving costs, partial feedback,
  per-window capacity, shadow-price pacing, and abrupt concept drift.

- Implemented contextual Greedy, LinUCB, and batched Discounted LinUCB policies with
  leakage-safe 49-dimensional features and a capacity-constrained clairvoyant oracle;
  verified routed-only learning, reproducibility, and zero capacity violations with
  automated tests.

- Evaluated 45 paired policy runs across three seeds and three capacity levels;
  LinUCB improved mean net profit by 17.3% over Random at 40% capacity and reduced
  pseudo-regret by 78.8%, while discount-factor sweeps showed that aggressive
  forgetting hurt performance after drift.

## Accuracy note

These are semi-synthetic experimental results, not production uplift. Keep words such
as "simulator," "Avazu contexts," and "paired policy runs" when adapting the bullets.
