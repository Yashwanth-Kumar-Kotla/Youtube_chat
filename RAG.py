from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

yt = YouTubeTranscriptApi()

video_id = 'KtHdbzChVWo' #just the id not the url

try: 
    transcript_list = yt.fetch(video_id, languages=["en"])
    transcript = " ".join(chunk['text'] for chunk in transcript_list.to_raw_data())
except TranscriptsDisabled:
    print("transcripts are not available")

splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)

chunks = splitter.create_documents([transcript])
print(len(chunks))
print(chunks[0:10])
print("---------------------")
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

vector_store = FAISS.from_documents(documents=chunks, embedding = embeddings)

retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {"k" : 4} )

question = 'go through the roadmap with me what should i start learning to become AI enginner how should i proceed?'

prompt = PromptTemplate(
    template='You are an helpful assistant, answer me to the following question using only the context ill provided. DONOT answer on your own ideas. Say you dont know the answer if the answer is not in the context. here is the context: {context}, and the question is, question: {question}',
    input_variables=["context", "question"]
)

contexts = retriever.invoke(question)

context_final = '/n/n'.join(context.page_content for context in contexts)
print("--------context_final------------------")
print(context_final)

final_prompt = prompt.invoke({"context": context_final, "question" : question})

model = ChatOpenAI()

result = model.invoke(final_prompt)
print("--------------result---------")
print(result.content)



