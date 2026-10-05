[![CI](https://github.com/dh-kt/auto-mpg-knn/actions/workflows/ci.yml/badge.svg)](https://github.com/dh-kt/auto-mpg-knn/actions/workflows/ci.yml)

# Vehicle Fuel Efficiency Prediction Pipeline 
An end-to-end machine learning project that predicts vehicle fuel efficiency (MPG) using the ISL Auto dataset. The project includes exploratory data analysis, preprocessing, model training, hyperparameter tuning, API deployment through FastAPI, Docker containerization, automated testing, and GitHub Actions CI.

Technical highlights 
Trained a distance‑weighted KNN (best_k = 21); test RMSE ≈ 4.10, R² ≈ 0.67.
Reproducible pipeline: 
1. cleaned Colab notebook, saved artifacts (scaler.joblib, knn_weighted.joblib), and model_data/summary.json.

2. Production-ready serving: FastAPI endpoint (/predict) validated locally and packaged with Docker.

3. Continuous integration: GitHub Actions workflow runs pytest with CI-friendly dummy artifacts.

4. Release assets: model binaries uploaded to GitHub Releases with SHA256 checksums for integrity.

Project Goal
Develop and deploy a machine learning pipeline capable of predicting vehicle fuel efficiency (MPG) based on vehicle characteristics such as displacement, horsepower, weight, acceleration, and cylinder count.

Quickstart (local):
1. Build image: `docker build -t auto-mpg-knn:latest .`
2. Run container (mount model_data): `docker run --rm -p 8000:8000 -v "<ABS_PATH>/model_data:/content/model_data" auto-mpg-knn:latest`
3. Test:
   - GET `http://127.0.0.1:8000/`
   - POST `http://127.0.0.1:8000/predict` with JSON body:
     `{"displacement":150,"horsepower":95,"weight":2000,"acceleration":15.5,"cylinders":4}`

Notes:
- Consider adding `model_data/` to `.gitignore` if prefer to not commit model binaries.
- See `notebooks/01_auto_knn_clean.ipynb` for the cleaned analysis.

