[![CI](https://github.com/dh-kt/auto-mpg-knn/actions/workflows/ci.yml/badge.svg)](https://github.com/dh-kt/auto-mpg-knn/actions/workflows/ci.yml)

# Vehicle Fuel Efficiency Prediction Pipeline 
**Python | Scikit-Learn | FastAPI | Docker | GitHub Actions**

Built a reproducible machine learning pipeline to predict vehicle fuel efficiency (MPG) using the Auto MPG dataset.

The best-performing distance-weighted KNN model (best_k = 21) achieved a test RMSE of approximately 4.10 and an R² score of approximately 0.67.

## Technical highlights

The project includes the following production-ready components:

1. Cleaned analysis notebook, saved artifacts (scaler.joblib, knn_weighted.joblib), and model_data/summary.json.

2. Production-ready serving: FastAPI endpoint (/predict) validated locally and packaged with Docker.

3. Continuous integration: GitHub Actions workflow runs pytest with CI-friendly dummy artifacts.

4. Release assets: model binaries uploaded to GitHub Releases with SHA256 checksums for integrity.

## Project Goal

Develop and deploy a machine learning pipeline capable of predicting vehicle fuel efficiency (MPG) based on vehicle characteristics such as displacement, horsepower, weight, acceleration, and cylinder count.

## Quickstart (local):
1. Build image: `docker build -t auto-mpg-knn:latest .`
2. Run container (mount model_data): `docker run --rm -p 8000:8000 -v "<ABS_PATH>/model_data:/content/model_data" auto-mpg-knn:latest`
3. Test:
   - GET `http://127.0.0.1:8000/`
   - POST `http://127.0.0.1:8000/predict` with JSON body:
     `{"displacement":150,"horsepower":95,"weight":2000,"acceleration":15.5,"cylinders":4}`

## Notes:
- Consider adding `model_data/` to `.gitignore` if prefer to not commit model binaries.
- See `notebooks/01_auto_knn_clean.ipynb` for the cleaned analysis.

