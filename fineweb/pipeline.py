from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter
import json

pipeline_exec = LocalPipelineExecutor(
    pipeline=[
        # replace "data/CC-MAIN-2024-10" with "sample/100BT" to use the 100BT sample
        ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10", limit=1000),
        LambdaFilter(lambda doc: "hugging" in doc.text),
        JsonlWriter("fineweb-dataset")
    ],
    tasks=10
)
pipeline_exec.run()



input_file = "fineweb-dataset.jsonl"
output_file = "fineweb-dataset.txt"

with open(input_file, "r") as jsonl, open(output_file, "w", encoding="utf-8") as txt:
    for line in jsonl:
        doc = json.loads(line)
        txt.write(doc["text"] + "\n\n")  # Separate documents with a newline

print("Conversion completed! Dataset saved as fineweb-dataset.txt")
