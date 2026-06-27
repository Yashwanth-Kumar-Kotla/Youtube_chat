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
output_parser = StrOutputParser()
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
model = ChatOpenAI()

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

prompt = PromptTemplate(
    template="You are a helpful assistant. Answer the following question using ONLY the provided context. "
              "Do NOT use your own knowledge. If the answer isn't in the context, say you don't know.\n\n"
              "Context: {context}\n\nQuestion: {question}",
    input_variables=["context", "question"]
)

# session_state holds the chain across reruns
if "main_chain" not in st.session_state:
    st.session_state.main_chain = None

video_id = st.text_input("Enter the YouTube Video ID -- DO NOT ENTER WHOLE URL")

if st.button("Submit") and video_id:
    try:
        transcripts_list = yt_api.fetch(video_id)
        transcripts = ' '.join(t['text'] for t in transcripts_list.to_raw_data())
    except TranscriptsDisabled:
        st.error("Transcripts for this video are disabled. Try another video.")
        st.stop()
    except Exception as e:
        st.error(f"Couldn't fetch transcript: {e}")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([transcripts])

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {"k" : 4} )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    st.session_state.main_chain = parallel_chain | prompt | model | output_parser
    st.success("Transcript processed. You can now ask questions.")

# This block only shows once a chain exists in session_state
if st.session_state.main_chain:
    question = st.text_input("Enter your question.")
    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            result = st.session_state.main_chain.invoke(question)
        st.write(result)