from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

Document = [
    "USA Visa: Requirements include valid passport, DS-160 form, visa fee payment receipt, and interview at US embassy. Process involves filling form online, scheduling interview, and attending with documents.",
    "Canada Visa: Requirements include passport, photo, proof of funds, travel history, and biometrics. Process involves applying online, paying fees, and giving biometrics.",
    "UK Visa: Requirements include passport, TB test (if applicable), financial proof, and CAS letter for study visa. Process involves online application, fee payment, and attending VFS appointment.",
    "Australia Visa: Requirements include passport, health insurance, GTE statement, and proof of funds. Process involves online application on ImmiAccount and medical exam.",
    "Schengen Visa: Requirements include passport, travel insurance, flight booking, hotel booking, and proof of funds. Process involves submitting documents at embassy and attending interview."
]

query = "What are the requirements and process for obtaining a US visa?"

doc_embedding = embedding.embed_documents(Document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

Index, Score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(Document[Index])
print("The similarity score is", Score)