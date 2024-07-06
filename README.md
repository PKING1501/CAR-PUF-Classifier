# README: Vulnerability Analysis of Companion Arbiter PUF (CAR-PUF)

## Objective

The objective of this document is to demonstrate the vulnerability of the Companion Arbiter PUF (CAR-PUF) to a single linear model, contrary to Melbo’s belief. We achieve this by simplifying the CAR-PUF into two linear models and analyzing their behavior.

## Approach

We proceed systematically, starting from basic principles and gradually advancing towards our goal:

- **Decomposition**: Decompose the CAR-PUF into two linear models, denoted as (u, p) and (v, q), which accurately predict its outputs.
- **Simplification**: By simplifying the CAR-PUF into these linear models, we lay the groundwork for our subsequent analysis.
- **Mapping Function**: Our mapping function, φ(c), depends solely on the challenge vector, c, and universal constants.
- **Step-by-Step Analysis**: Proceed step by step, considering necessary transformations and adjustments, to arrive at a linear model with a different dimensionality compared to the individual 32-bit models.
- **Mathematical Derivation**: Through rigorous mathematical derivation, define the mapping function φ(c) and determine the dimensionality of the feature vector, W, required for the linear model.

## Principles

Our approach adheres to the principles outlined in the problem statement:

- Aligning with given constraints and objectives.
- Incorporating insights from discussion hours and lectures.

## Conclusion

In conclusion, this README outlines our methodical approach to demonstrate the vulnerability of the CAR-PUF to a single linear model. The decomposition into two linear models (u, p) and (v, q) highlights our findings, supported by rigorous mathematical derivation and adherence to specified principles.
