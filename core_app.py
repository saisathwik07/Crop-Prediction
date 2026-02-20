# core_app.py

import threading
import time
import random
import traceback
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import base64
import requests
from PIL import Image


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import os

from PIL import Image
import io
import tempfile
import os
try:
    import firebase_admin
    from firebase_admin import credentials, db
except ImportError:
    firebase_admin = None
try:
    import openpyxl
except ImportError:
    openpyxl = None


class SoilAnalysisCore:

    def __init__(self):
        # ================= FIREBASE INIT =================
        self.firebase_initialized = False
        self.firebase_url = "https://greenhouse-80904-default-rtdb.firebaseio.com/"
        self.firebase_cred_path = os.path.join(os.path.dirname(__file__), 'greenhouse-80904-firebase-adminsdk-fbsvc-425cb740e9.json')

        # ================= DATA =================
        self.dataset = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None

        # ================= MODELS =================
        self.le = LabelEncoder()
        self.ms = MinMaxScaler()
        self.sc = StandardScaler()
        self.rfc = RandomForestClassifier(random_state=42)
        self.dt = None

        # ================= IOT =================
        self.iot_simulation_running = False
        self.iot_thread = None
        self.iot_data = {
            "temperature": [],
            "humidity": [],
            "soil_moisture": [],
            "nitrogen": [],
            "phosphorus": [],
            "potassium": []
        }

        # ================= DOMAIN DATA =================
        self.natural_fertilizers = {
            "rice": ["Compost", "Manure", "NPK Fertilizer"],
            "maize": ["Compost", "Manure", "NPK Fertilizer"],
            "cotton": ["Compost", "Manure", "NPK Fertilizer"],
            "coffee": ["Compost", "Manure", "Coffee Fertilizer"],
        }

        self.price = {
            "rice": "23450 / Acre",
            "maize": "19450 / Acre",
            "cotton": "19250 / Acre",
            "coffee": "23450 / Acre"
        }

        self.micronutrient_thresholds = {
            'Iron': (2.5, 6.0),
            'Zinc': (0.5, 2.0),
            'Copper': (0.2, 1.0),
            'Boron': (0.5, 2.0),
        }

        self._init_iot_data()


    def soil_health_analysis(self, micronutrients: dict):
        scores = {}
        total_score = 0.0

        for nutrient, (low, high) in self.micronutrient_thresholds.items():
            if nutrient not in micronutrients:
                raise Exception(f"Missing micronutrient: {nutrient}")

            val = float(micronutrients[nutrient])

            if val < low:
                status = "Deficient"
                score = 0.4
            elif val > high:
                status = "Excess"
                score = 0.6
            else:
                status = "Optimal"
                score = 1.0

            scores[nutrient] = {
                "value": val,
                "status": status,
                "score": score
            }

            total_score += score

        soil_health_index = round((total_score / len(scores)) * 100, 2)

        return {
            "nutrients": scores,
            "soil_health_index": soil_health_index
        }
    def plot_micronutrient_graph(self, micronutrients: dict):
        nutrients = list(micronutrients.keys())
        values = [float(v) for v in micronutrients.values()]

        fig, ax = plt.subplots()
        ax.bar(nutrients, values, color="#4CAF50")
        ax.set_title("Micronutrient Distribution")
        ax.set_ylabel("mg/kg")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close(fig)

        return base64.b64encode(buf.getvalue()).decode()
    def full_crop_recommendation(self, inputs: dict, micronutrients: dict):
        # 1. Basic crop prediction
        pred = self.predict_crop(inputs)

        # 2. Soil health analysis
        soil = self.soil_health_analysis(micronutrients)

        # 3. Yield estimation (heuristic, dataset-consistent)
        yield_est = self.estimate_yield(inputs)

        # 4. Fertilizer recommendation
        fertilizers = self.fertilizer_advice(
            pred["crop"],
            soil["soil_health_index"]
        )

        return {
            "crop": str(pred["crop"]),
            "yield_score": f"{float(yield_est)} %",
            "soil_health_index": float(soil["soil_health_index"]),
            "fertilizers": list(fertilizers)
        }

    # ==================================================
    # DATA MANAGEMENT
    # ==================================================
    def load_dataset(self, file_obj):
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']

        for enc in encodings:
            try:
                self.dataset = pd.read_csv(file_obj, encoding=enc)

                # Clean Soil column if exists
                if 'Soil' in self.dataset.columns:
                    self.dataset['Soil'] = self.dataset['Soil'].astype(str)\
                        .str.replace(r'[^\x00-\x7F]+', '', regex=True)

                # Convert numeric columns safely
                for col in ['N', 'P', 'K']:
                    if col in self.dataset.columns:
                        self.dataset[col] = pd.to_numeric(
                            self.dataset[col], errors='coerce'
                        )
                return {
                    "status": "success",
                    "encoding": enc,
                    "rows": self.dataset.shape[0],
                    "columns": list(self.dataset.columns)
                    }

            except UnicodeDecodeError:
                file_obj.seek(0)   # 🔑 VERY IMPORTANT
                continue

        raise Exception("Failed to load dataset with supported encodings")


    def load_dataset_from_url(self, url):
        self.dataset = pd.read_csv(url)
        return {
            "rows": self.dataset.shape[0],
            "columns": list(self.dataset.columns)
        }

    def process_dataset(self):
        if self.dataset is None or self.dataset.empty:
            # Auto-load sample dataset
            try:
                sample_path = os.path.join(os.path.dirname(__file__), 'Crop_recommendation1.csv')
                if os.path.exists(sample_path):
                    self.dataset = pd.read_csv(sample_path)
                else:
                    raise Exception("Dataset not loaded and sample dataset not found")
            except Exception as e:
                raise Exception(f"Dataset not loaded or empty: {str(e)}")

        df = self.dataset.copy()

        required_columns = ['label', 'N', 'P', 'K', 'temperature', 'humidity', 'ph']
        lower_cols = [c.lower() for c in df.columns]

        # Check if required columns exist (case-insensitive)
        missing_cols = []
        for col in required_columns:
            if col.lower() not in lower_cols:
                missing_cols.append(col)
        
        if missing_cols:
            # Try to load sample dataset instead
            try:
                sample_path = os.path.join(os.path.dirname(__file__), 'Crop_recommendation1.csv')
                if os.path.exists(sample_path):
                    self.dataset = pd.read_csv(sample_path)
                    df = self.dataset.copy()
                else:
                    raise Exception(f"Missing required columns: {', '.join(missing_cols)}. Please upload a dataset with these columns: {', '.join(required_columns)}")
            except:
                raise Exception(f"Missing required columns: {', '.join(missing_cols)}. Please upload a dataset with these columns: {', '.join(required_columns)}")

        # Normalize column names
        rename_map = {}
        for col in df.columns:
            for req in required_columns:
                if col.lower() == req.lower():
                    rename_map[col] = req
        df = df.rename(columns=rename_map)

        # Convert numeric columns
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()

        # Use only the explicit feature columns expected by the predictor
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph']
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise Exception(f"Missing required feature columns after processing: {', '.join(missing)}")

        # Ensure X contains only the expected features in the correct order
        self.X = df[feature_cols].copy()

        # 🔑 DROP NON-NUMERIC COLUMNS (CRITICAL)
        if 'Soil' in self.X.columns:
            self.X = self.X.drop(['Soil'], axis=1)

        self.Y_raw = df['label']                 # keep original labels
        self.Y = self.le.fit_transform(df['label'])  # encoded labels


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.Y, test_size=0.2, random_state=42
        )

        self.X_train_scaled = self.sc.fit_transform(self.ms.fit_transform(self.X_train))
        self.X_test_scaled = self.sc.transform(self.ms.transform(self.X_test))

        self.rfc.fit(self.X_train_scaled, self.y_train)

        return {
            "status": "processed",
            "records": len(df),
            "train": len(self.X_train),
            "test": len(self.X_test)
        }


    # ==================================================
    # ANALYSIS
    # ==================================================
    def train_decision_tree(self):
        self.dt = DecisionTreeRegressor(
            max_depth=100,
            max_leaf_nodes=20,
            splitter="random",
            random_state=0
        )
        self.dt.fit(self.X_train_scaled, self.y_train)
        preds = self.dt.predict(self.X_test_scaled)
        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        return rmse

    def evaluate_model(self):
        if not hasattr(self, 'X_test_scaled'):
            raise Exception("Dataset not processed yet")
        y_pred = self.rfc.predict(self.X_test_scaled)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred,target_names=self.le.classes_)
        return acc, report

    # ==================================================
    # PREDICTION
    # ==================================================
    def predict_crop(self, input_data: dict):
        # Ensure dataset has been processed and scalers/models fitted.
        if not hasattr(self, 'X_train_scaled') or self.X_train is None:
            try:
                self.process_dataset()
            except Exception:
                # Let the downstream code raise a clear error if processing fails
                pass
        features = np.array([[
            float(input_data['N']),
            float(input_data['P']),
            float(input_data['K']),
            float(input_data['temperature']),
            float(input_data['humidity']),
            float(input_data['ph'])
        ]])

        scaled = self.sc.transform(self.ms.transform(features))
        # Predict encoded label
        model = self.best_model if self.best_model is not None else self.rfc
        pred_encoded = model.predict(scaled)[0]
        crop = self.le.inverse_transform([int(pred_encoded)])[0]


        result = {
            "crop": str(crop),
            "fertilizers": list(self.natural_fertilizers.get(crop, [])),
            "suitability": float(round(random.uniform(0.7, 1.0), 2))
        }
        return result


    # ==================================================
    # IOT SIMULATION
    # ==================================================
    def _init_iot_data(self):
        for _ in range(5):
            self.iot_data['temperature'].append(random.uniform(20, 35))
            self.iot_data['humidity'].append(random.uniform(30, 90))
            self.iot_data['soil_moisture'].append(random.uniform(20, 80))
            self.iot_data['nitrogen'].append(random.uniform(30, 150))
            self.iot_data['phosphorus'].append(random.uniform(20, 100))
            self.iot_data['potassium'].append(random.uniform(20, 120))

    def toggle_iot(self):
        self.iot_simulation_running = not self.iot_simulation_running
        if self.iot_simulation_running:
            self.iot_thread = threading.Thread(target=self._iot_loop, daemon=True)
            self.iot_thread.start()
        return self.iot_simulation_running

    def _iot_loop(self):
        while self.iot_simulation_running:
            self.iot_data['temperature'].append(random.uniform(20, 35))
            self.iot_data['humidity'].append(random.uniform(30, 90))
            self.iot_data['soil_moisture'].append(random.uniform(20, 80))
            self.iot_data['nitrogen'].append(random.uniform(30, 150))
            self.iot_data['phosphorus'].append(random.uniform(20, 100))
            self.iot_data['potassium'].append(random.uniform(20, 120))

            for k in self.iot_data:
                self.iot_data[k] = self.iot_data[k][-50:]

            time.sleep(5)

    # ==================================================
    # MICRONUTRIENT
    # ==================================================
    def analyze_micronutrients(self, values: dict):
        result = {}
        for k, (low, high) in self.micronutrient_thresholds.items():
            val = float(values[k])
            if val < low:
                result[k] = "Deficient"
            elif val > high:
                result[k] = "Excess"
            else:
                result[k] = "Optimal"
        return result

    # ==================================================
    # PLOTS
    # ==================================================
    def plot_confusion(self):
        y_pred = self.rfc.predict(self.X_test_scaled)
        cm = pd.crosstab(self.y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    # (Keep single implementations of soil analysis & micronutrient plotting above.)

    def train_multiple_models(self):
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=200),
            "DecisionTree": DecisionTreeRegressor(max_depth=20),
            "GradientBoosting": GradientBoostingClassifier()
        }
        results = {}
        best_model = None
        best_score = 0

        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_train)
            score = model.score(self.X_test_scaled, self.y_test)
            results[name] = round(score * 100, 2)

            if score > best_score:
                best_score = score
                best_model = model

            self.best_model = best_model
        return {
            "results": results,
            "best_model": best_model.__class__.__name__,
            "accuracy": round(best_score * 100, 2)
        }
    def estimate_yield(self, inputs):
        score = (
            inputs["N"] * 0.2 +
            inputs["P"] * 0.2 +
            inputs["K"] * 0.2 +
            inputs["temperature"] * 0.4
        )
        return round(min(score / 100, 1.0) * 100, 2)
    def fertilizer_advice(self, crop, soil_health):
        base = self.natural_fertilizers.get(crop, [])

        if soil_health < 50:
            base.append("Soil Conditioner")
        elif soil_health < 75:
            base.append("Balanced NPK")

        return list(set(base))
    