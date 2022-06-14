"""A Gradio app for searching similar texts.
"""

import gradio as gr
import numpy as np

from finegrained.models import embed
from finegrained.utils.data import load_data
from finegrained.utils.huggingface_transformers import SentenceEmbeddings
from finegrained.utils.similarity import SimilaritySearch

MODELS = {
    "text_similarity": ["symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"]
}


def _greet(name):
    return "Hello " + name + "!!"


def greet():
    demo = gr.Interface(fn=_greet, inputs="text", outputs="text")
    demo.launch()


def _embed_text(file, text_field, model_name):
    # TODO validate input
    data = load_data(file.name)
    embeddings = np.random.rand(len(data), 768)
    model = SentenceEmbeddings(model_name)
    return data.to_dict("records"), embeddings, model


def _get_chatbot_answer(text_input, history, model, data, embeddings):
    sim = SimilaritySearch(data, embeddings)
    top_sim = sim.embed_and_find_similar(text_input, model, top_k=5)

    # TODO distance threshold
    answer = top_sim[0]["answer"]
    history.append((text_input, answer))

    return history, ""


def chatbot():
    models = MODELS["text_similarity"]
    fields = ["question", "answer"]

    demo = gr.Blocks()
    with demo:
        gr.Markdown("# Retrieval chatbot")
        gr.Markdown("Upload your data and starting chatting")
        with gr.Tabs():
            with gr.TabItem("Upload data"):
                gr.Markdown(
                    f"Upload a CSV file with the {fields} "
                    "columns, select field and model and submit."
                )

                with gr.Box():
                    file = gr.File(label="CSV file")
                    text_field = gr.Radio(
                        choices=fields,
                        value=fields[0],
                        label="Which field to use",
                    )
                    model_name = gr.Dropdown(
                        choices=models,
                        value=models[0],
                        label="Which model to use",
                    )
                    file_btn = gr.Button("Submit")

                data = gr.JSON(
                    label="My data",
                )
                embeddings = gr.Variable()
                model = gr.Variable()

            with gr.TabItem("Chat"):
                chatbot = gr.Chatbot(label="Dialog")
                with gr.Row():
                    with gr.Box():
                        text_input = gr.Textbox(
                            lines=2,
                            max_lines=10,
                            placeholder="Enter a message",
                            label="Message",
                        )
                        text_btn = gr.Button("Send")

        file_btn.click(
            _embed_text,
            inputs=[file, text_field, model_name],
            outputs=[data, embeddings, model],
        )
        text_btn.click(
            _get_chatbot_answer,
            inputs=[text_input, chatbot, model, data, embeddings],
            outputs=[chatbot, text_input],
        )

    demo.launch(debug=True, inbrowser=True)
