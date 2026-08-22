# Security Policy

## Scope

This project is a machine learning research repository. The main risks are accidental exposure of credentials, dataset files, local paths, and generated artifacts that were not meant to be public.

## Do Not Commit

- Kaggle API credentials
- `.env` files or tokens
- Raw downloaded datasets
- Local cache directories
- Personal machine paths
- Large model artifacts unless intentionally documented

## Reporting a Concern

Report sensitive data exposure privately to the repository owner. Include the affected path, the type of data, and whether it appears in the latest commit or repository history.