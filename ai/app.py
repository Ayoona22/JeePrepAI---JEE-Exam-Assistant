from flask import Flask, request, jsonify, Response
import google.generativeai as genai
from transformers import AutoTokenizer
from PIL import Image
import io, os, time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 30000))

BASE_PROMPT = """
You are a JEE assistant.

Instructions:
- Answer only questions related to Chemistry, Physics, Mathematics and JEE.
- Prioritize using the study materials provided.
- If the study materials are not relevant, rely on your own knowledge.
- Use precise and helpful explanations. If the user prefers, answer briefly.
- Provide formulas, equations, and step-by-step logic when needed.
- Do **not** use LaTeX. Just use plain text formatting.
- For example, write `C6H6 + Br2 → C6H5Br + HBr` instead of LaTeX code like `\\text{{C}}_6\\text{{H}}_6 + \\text{{Br}}_2 → ...`.

Use the following:
1. Study Material:
{study_material}

2. Chat Summary:
{chat_summary}

3. Chat History:
{chat_history}

4. Current Question:
{user_question}
"""

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        input_type = data.get("input_type", "text")
        file_data = data.get("file_data")

        # Compose prompt
        user_question = data.get("user_input", "")
        prompt = BASE_PROMPT.format(
            study_material="\n".join(data.get("study_material", [])),
            chat_summary=data.get("chat_summary", ""),
            chat_history=data.get("chat_history", ""),
            user_question=user_question
        )

        # Count tokens
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > MAX_TOKENS:
            prompt = f"""
            Please summarize this overlong prompt for a JEE assistant into a short, complete context:
            {prompt}
            """
            response = model.generate_content(prompt)
            prompt = response.text.strip()

        # IMAGE MODE
        if input_type == "image" and file_data:
            image_bytes = bytes.fromhex(file_data["bytes"])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            content = [
                image,
                f"""
                You are a JEE assistant.
                The uploaded image contains a JEE question. Use the image to understand the question directly.
                Instructions:
                - Answer only questions related to Chemistry, Physics, Mathematics and JEE.
                - Prioritize using the study materials provided.
                - If the study materials are not relevant, rely on your own knowledge.
                - Use precise and helpful explanations. If the user prefers, answer briefly.
                - Provide formulas, equations, and step-by-step logic when needed.
                - Do **not** use LaTeX. Just use plain text formatting.
                - For example, write C6H6 + Br2 → C6H5Br + HBr instead of LaTeX code like \\text{{C}}_6\\text{{H}}_6 + \\text{{Br}}_2 → ....


                Use the following:
                - Study Material:
                {data.get("study_material", [])}

                - Summary of previous conversation:
                {data.get("chat_summary", "")}
                - Previous chats:
                {data.get("chat_history", "")}
                - Current Question:
                {data.get("user_input", "")}


                Please respond with a clear, correct answer to the question shown in the image. 
                Do not mention that it's an image. Do not restate the question.
                Return only the final answer.
                """
            ]
            response = model.generate_content(content)
            result_text = response.text.strip()
        else:
            response = model.generate_content(prompt)
            result_text = response.text.strip()

        def generate_stream():
            for char in result_text:
                yield char
                time.sleep(0.002)

        return Response(generate_stream(), content_type="text/plain")

    except Exception as e:
        return Response(f"Error: {str(e)}", content_type="text/plain"), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        prev = data.get("previous_summary", "")
        new_dialogue = data.get("new_dialogue", "")
        prompt = f"""
Summarize the following conversation between a JEE assistant and a student in under 100 words.

Previous Summary:
{prev}

New Dialogue:
{new_dialogue}
"""
        response = model.generate_content(prompt)
        return jsonify({"summary": response.text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🧠 AI Service is running at http://localhost:5003")
    app.run(host="0.0.0.0", port=5003, debug=True)
