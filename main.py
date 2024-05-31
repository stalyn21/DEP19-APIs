from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from io import BytesIO

app = FastAPI()

def limpiarDatos(df):
    df_limpio = df.dropna()
    return df_limpio

def preprocesarDatos(df, test_size, val_size, random_state, normalize):
    features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP',
                'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name']
    target = 'Transported'

    X = df[features]
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    categorical_features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))])
    
    if normalize:
        numeric_transformer.steps.append(('scaler', StandardScaler()))

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor, X_train, X_val, X_test, y_train, y_val, y_test

def entrenarYGuardarModelo(preprocessor, X_train, X_val, X_test, y_train, y_val, y_test, model_path, random_state):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=random_state))])

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f'Train Accuracy: {train_accuracy:.2f}')

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy: {val_accuracy:.2f}')

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.2f}')

    joblib.dump(model, model_path)
    print(f'Modelo guardado en {model_path}')

    return model, train_accuracy, val_accuracy, test_accuracy

@app.post("/train")
async def train_model(
    random_state: int = Form(...), 
    test_size: float = Form(...), 
    val_size: float = Form(...), 
    normalize: bool = Form(...), 
    save_path: str = Form("./spaceship_titanic_model.pkl")):

    try:
        df = pd.read_csv("train.csv")

        df_limpio = limpiarDatos(df)

        preprocessor, X_train, X_val, X_test, y_train, y_val, y_test = preprocesarDatos(df_limpio, test_size, val_size, random_state, normalize)
        model, train_accuracy, val_accuracy, test_accuracy = entrenarYGuardarModelo(
            preprocessor, X_train, X_val, X_test, y_train, y_val, y_test, save_path, random_state
        )

        return {
            "message": "Model trained and saved successfully.",
            "model_path": save_path,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
