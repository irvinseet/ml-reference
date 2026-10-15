### Production Readiness Considerations

- **Data leakage:** Ensure preprocessing is fitted only on training data and exclude features unavailable at prediction time (e.g., call duration).
- **Model validation:** Use cross-validation for model selection and an untouched test set for final evaluation.
- **Business alignment:** Evaluate Precision@K to reflect the bank's limited calling capacity.
- **Inference pipeline:** Reuse the same preprocessing and feature transformations during training and prediction.
- **Monitoring:** In a production deployment, monitor feature drift, prediction distributions, and actual conversion rates once outcomes become available.

**Status:** Exploratory ML project. Production deployment and monitoring have not been implemented.
