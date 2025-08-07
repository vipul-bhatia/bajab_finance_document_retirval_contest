
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def regenerate_answers():
    """
    Reads questions_answers.jsonl, sends requests to the API,
    and saves the new results to new_questions_answers.jsonl.
    """
    input_file = "previous_questions_answers.jsonl"
    output_file = "new_questions_answers.jsonl"
    api_url = "http://localhost:8000/api/v1/hackrx/run"
    token = os.getenv("EXPECTED_TOKEN")

    if not token:
        print("Error: EXPECTED_TOKEN not found in .env file.")
        return

    headers = {"Authorization": f"Bearer {token}"}

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                document_url = data.get("document_url")
                questions = data.get("questions")

                if not document_url or not questions:
                    print(f"Skipping invalid entry: {data}")
                    continue

                payload = {
                    "documents": document_url,
                    "questions": questions
                }

                print(f"Processing document: {document_url}")
                response = requests.post(api_url, json=payload, headers=headers)

                if response.status_code == 200:
                    new_answers = response.json().get("answers")
                    new_entry = {
                        "timestamp": data.get("timestamp"),
                        "document_name": data.get("document_name"),
                        "document_url": document_url,
                        "questions": questions,
                        "answers": new_answers
                    }
                    f_out.write(json.dumps(new_entry) + '\n')
                    print("Successfully generated new answers.")
                else:
                    print(f"Error processing entry: {data}")
                    print(f"Status code: {response.status_code}")
                    print(f"Response: {response.text}")

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    regenerate_answers()
