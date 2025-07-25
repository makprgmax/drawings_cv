import pandas as pd
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import openpyxl


connections.connect("default", host="", port="")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=500)
]
schema = CollectionSchema(fields, description="Employee task analysis")
collection_name = "employee_tasks"
if collection_name not in Collection.list_collections():
    collection = Collection(collection_name, schema)
else:
    collection = Collection(collection_name)

data = pd.read_csv("data_csv.csv", sep=";")


data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y %H:%M:%S")

model = SentenceTransformer('all-MiniLM-L6-v2') 
text_data = data["Project"] + " " + data["View"] + " " + data["Action"]
vectors = model.encode(text_data.tolist())


metadata = text_data.tolist()
entities = [
    {"embedding": vector.tolist(), "metadata": meta}
    for vector, meta in zip(vectors, metadata)
]
collection.insert([list(range(len(entities))),  # IDs
                   [e["embedding"] for e in entities],  # Векторы
                   [e["metadata"] for e in entities]])  # Метаданные
collection.load()


def analyze_tasks(data, start_time, end_time):
    filtered_data = data[(data["Date"] >= start_time) & (data["Date"] <= end_time)]

    grouped = filtered_data.groupby("Employee")

    report = []
    for user, group in grouped:
        total_tasks = len(group)
        
        modeling_tasks = group["Sheets"].isna().sum()
        drawings_tasks = total_tasks - modeling_tasks
        modeling_percent = (modeling_tasks / total_tasks) * 100
        drawings_percent = (drawings_tasks / total_tasks) * 100

        if abs(modeling_percent - drawings_percent) <= 10:
            main_task = "Modeling/Drawings"
        elif modeling_percent > drawings_percent:
            main_task = "Modeling"
        else:
            main_task = "Drawings"


        report.append({
            "Период времени": f"{start_time} - {end_time}",
            "Основная задача": main_task,
            "Modeling %": round(modeling_percent, 2),
            "Drawings %": round(drawings_percent, 2),
            "User": user
        })
    return report

start_time = datetime(2025, 1, 14, 22, 1, 0) 
end_time = datetime(2025, 1, 14, 22, 2, 0)   

report_data = analyze_tasks(data, start_time, end_time)

