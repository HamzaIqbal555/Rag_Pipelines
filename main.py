import warnings
import nltk
from transformers import pipeline, AutoTokenizer
import pdfplumber
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

pdf_path = "./google_terms_of_service_en_in.pdf"

output_txt_file = 'extracted_txt_file'

# extracting text from pdf to text file
with pdfplumber.open(pdf_path) as pdf:
    extracted_text = ""
    for page in pdf.pages:
        extracted_text += page.extract_text()

with open(output_txt_file, "w") as text_file:
    text_file.write(extracted_text)


# reading text file
with open(output_txt_file, "r") as file:
    document_text = file.read()

# loading the summarization pipeline
summarizer = pipeline('summarization', model='t5-small')

# summarizing the small text (since the text file is too large)
summary = summarizer(
    document_text[:1000], max_length=150, min_length=30, do_sample=False)

# print(summary[0]['summary_text'])

passages = []
current_passage = ''

sentences = nltk.sent_tokenize(document_text)

for sentence in sentences:
    if len(current_passage.split()) + len(sentence.split()) < 200:
        current_passage += ' ' + sentence

    else:
        passages.append(current_passage.strip())
        current_passage = sentence

if current_passage:
    passages.append(current_passage.strip())

# print(passages)

tokenizer = AutoTokenizer.from_pretrained(
    'valhalla/t5-base-qg-hl', legacy=False)
qg_pipeline = pipeline('text2text-generation',
                       model='valhalla/t5-base-qg-hl', device=-1, tokenizer=tokenizer)


def generate_questions(passage, min_questions=3):
    input_text = 'Generate questions:' + passage
    questions = qg_pipeline(
        input_text,
        num_return_sequences=min_questions,
        num_beams=5,
        max_new_tokens=50
    )
    # if len(questions) < min_questions:
    return [item['generated_text'] for item in questions]


# for i, passage in enumerate(passages):
#     questions = generate_questions(passage)
#     print(f"Passage {i+1}: {passage}")
#     print(f"Generated questions:")
#     for q_idx, q in enumerate(questions):
#         print(f"  {q_idx+1}. {q}")


qa_pipeline = pipeline('question-answering',
                       model='deepset/roberta-large-squad2', device=-1)

def generate_answers(passages, qa_pipeline):
    answered_questions = set()
    for i, passage in enumerate(passages):
        questions = generate_questions(passage)
        print(f'Passage: {i+1}')
        for question in questions:
            if question not in answered_questions:
                answer = qa_pipeline(
                    {'question': question, 'context': passage})
                print(f'Question: {question}')
                print(f'Answer: {answer['answer']}')

            answered_questions.add(question)


generate_answers(passages, qa_pipeline)
