from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
from typing import List
import io
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
import json
from datetime import datetime
from fastapi import Query 

app = FastAPI(title="Insurance Prediction API")

# Загрузка обученного пайплайна
model = joblib.load('model.pkl')  

# Структуры данных
class InsuranceData(BaseModel):
    Age: float
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: float
    Gender: str
    Vehicle_Age: str
    Vehicle_Damage: str

class BatchRequest(BaseModel):
    records: List[InsuranceData]

# Эндпоинты
@app.post("/predict_single")
async def predict_single(data: InsuranceData):
    try:
        # Преобразуем в DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Предсказание
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)
        
        return {
            "prediction": int(prediction[0]),
            "probability_class_0": float(proba[0][0]),
            "probability_class_1": float(proba[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(
        file: UploadFile = File(...), 
        show_sample: bool = Query(
        default=True,
        description="Показывать только первую запись в ответе для удобства просмотра в Swagger")):
    try:
        # Проверка формата файла
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        # Чтение данных
        contents = await file.read()
        input_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Проверка колонок
        required_cols = ['Age', 'Driving_License', 'Region_Code',
                        'Previously_Insured', 'Annual_Premium',
                        'Policy_Sales_Channel', 'Vintage',
                        'Gender', 'Vehicle_Age', 'Vehicle_Damage']
        
        if not all(col in input_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in input_df.columns]
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {', '.join(missing)}"
            )
        
        # Предсказание
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)
        
        # Добавляем результаты в DataFrame
        input_df['prediction'] = predictions
        input_df['probability_class_0'] = probabilities[:, 0]
        input_df['probability_class_1'] = probabilities[:, 1]
        
        # Преобразуем в список словарей
        results = input_df.to_dict(orient='records')

        # Возвращаем либо первую запись, либо все данные
        if show_sample:
            return JSONResponse({
                "message": "Showing first record only (use show_sample=false for full output)",
                "sample_record": results[0] if len(results) > 0 else {},
                "total_records": len(results)
            })
        else:
            return JSONResponse(results)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)