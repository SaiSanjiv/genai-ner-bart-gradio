# Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

## AIM: 
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BERT (as a BART alternative for public access) model and deploying the application using the Gradio framework for user interaction and evaluation.

## PROBLEM STATEMENT :
Named Entity Recognition (NER) is a fundamental Natural Language Processing (NLP) task that identifies and classifies entities such as names of people, organizations, locations, and miscellaneous entities in text.This project aims to create a simple, interactive web-based prototype that performs NER using a fine-tuned transformer model.The challenge lies in integrating a pre-trained transformer model with a user-friendly front-end to visualize entity recognition results efficiently.

## DESIGN STEPS:
### STEP 1:
* Import the required libraries — Transformers, Torch, and Gradio.
* Load a fine-tuned transformer model (BERT/BART) for the NER task using the Hugging Face model hub.

### STEP 2:
* Create a pipeline using the pipeline() function from Hugging Face to handle tokenization and entity recognition automatically.
* Define a function to process user input text and display recognized entities in a formatted structure.

### STEP 3:
* Design an interactive interface using Gradio with appropriate input and output components.
* Launch the interface locally to test and visualize the model’s predictions on custom text inputs.

## PROGRAM :
```
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import gradio as gr

model_name = "dslim/bert-large-NER"   
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def ner_function(text):
    entities = ner_pipeline(text)
    return "\n".join([f"{ent['word']} ({ent['entity_group']})" for ent in entities])

iface = gr.Interface(
    fn=ner_function,
    inputs=gr.Textbox(lines=5, label="Input Text"),
    outputs=gr.Textbox(lines=10, label="Named Entities"),
    title="NER – Using BERT Model + Gradio"
)

iface.launch()
```

## OUTPUT:
<img width="1919" height="839" alt="image" src="https://github.com/user-attachments/assets/93ca8b3e-096d-43a4-83d4-63a0e27f7e51" />

## RESULT:
A prototype Named Entity Recognition (NER) application was successfully developed using a fine-tuned transformer model and deployed via the Gradio framework.
The system accurately identifies and classifies entities such as people, organizations, and domains, demonstrating the capability of transformer-based models in real-world NLP applications.

