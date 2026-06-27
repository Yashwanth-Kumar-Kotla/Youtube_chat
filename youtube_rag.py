from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

yt_api = YouTubeTranscriptApi()

parser = StrOutputParser()

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

model = ChatOpenAI()

def formatstr(docs):
    return '/n/n'.join(doc.page_content for doc in docs)
    

video_id = st.text_input("Enter the Youtube Video ID -- DONOT ENTER WHOLE URL")
if st.button("Submit"):
    try:
        transcripts_list = yt_api.fetch(video_id)
        transcripts = ' '.join(transcript['text'] for transcript in transcripts_list.to_raw_data())
    except TranscriptsDisabled:
        st.write("Transcripts for this video is disabled, Try other video.")

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    chunks = splitter.create_documents([transcripts])
    

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()

    

    prompt = PromptTemplate(
        template="You are an helpful assistant, answer me to the following question using only the context ill provided. DONOT answer on your own ideas. Say you dont know the answer if the answer is not in the context. here is the context: {context}, and the question is, question: {question}",
        input_variables=["context", "question"]
    )
    parallel_chain = RunnableParallel({
        "context" : retriever | RunnableLambda(formatstr),
        "question" : RunnablePassthrough()
    })
    main_chain = parallel_chain | prompt | model | parser

question = st.text_input("Enter your Question.")
if st.button("Result"):
    result = main_chain.invoke(question)
    st.write(result)


     