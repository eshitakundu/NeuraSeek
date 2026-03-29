import os
import streamlit as st
from dotenv import load_dotenv
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.rag_chain import RAGChain
from streamlit_lottie import st_lottie

load_dotenv()

st.set_page_config(
    page_title="NeuraSeek",
    page_icon="https://raw.githubusercontent.com/eshitakundu/NeuraSeek/refs/heads/main/src/static/assistant.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://raw.githubusercontent.com/eshitakundu/NeuraSeek/refs/heads/main/src/static/bg.png');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>

    <audio autoplay loop>
        <source src="https://github.com/eshitakundu/NeuraSeek/raw/refs/heads/main/src/static/audio.mp3" type="audio/mpeg">
    </audio>
    """,
    unsafe_allow_html=True
)

# Initializing session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = set()  # to prevent duplicates
if "processing_file" not in st.session_state:
    st.session_state.processing_file = False
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# Initialize components
@st.cache_resource
def init_components():
    """Initialize RAG components."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        st.error("NVIDIA API Key not found! Please set it in .env file.")
        st.stop()
    
    vector_store = VectorStore(api_key)
    rag_chain = RAGChain(api_key)
    doc_processor = DocumentProcessor()
    
    # Try to load existing vector store
    try:
        vector_store.load()
    except:
        pass
    
    return vector_store, rag_chain, doc_processor

vector_store, rag_chain, doc_processor = init_components()
st.session_state.vector_store = vector_store
st.session_state.rag_chain = rag_chain

left_col, right_col = st.columns([0.30, 0.70])

with left_col:

    st.markdown(
    "<h2 style='text-align: center;'>Neura<span style='color:#ff9900ff;'>Seek</span></h2>",
    unsafe_allow_html=True
)
    with st.container(border=True):
        # File upload
        st.markdown("#### Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx'],
            help="Supported formats: PDF, DOCX, TXT, CSV, XLSX",
            key="file_uploader"
        )
        
        # Process file only if it's new and not currently processing
        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            # Check if this is a new file
            if (file_id not in st.session_state.documents_loaded and 
                not st.session_state.processing_file):
                
                st.session_state.processing_file = True
                
                # Progress indicator
                progress_bar = st.progress(0, text="Processing document...")
                status_text = st.empty()
                
                try:
                    # Save file temporarily
                    temp_path = f"data/temp_{uploaded_file.name}"
                    os.makedirs("data", exist_ok=True)
                    
                    progress_bar.progress(10, text="Saving file...")
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    progress_bar.progress(30, text="Extracting text...")
                    
                    # Process document
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    if file_type == 'doc':
                        file_type = 'docx'
                    
                    chunks = doc_processor.process_document(temp_path, file_type)
                    
                    progress_bar.progress(60, text=f"Creating embeddings for {len(chunks)} chunks...")
                    status_text.info(f"Processing {len(chunks)} chunks... This may take a minute.")
                    
                    # Add to vector store
                    vector_store.add_documents(chunks, uploaded_file.name)
                    
                    progress_bar.progress(90, text="Saving to disk...")
                    vector_store.save()
                    
                    # Mark as loaded
                    st.session_state.documents_loaded.add(file_id)
                    
                    progress_bar.progress(100, text="Complete!")
                    st.success(f"Successfully processed {len(chunks)} chunks from **{uploaded_file.name}**")
                    
                    # Clean up
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                finally:
                    st.session_state.processing_file = False
                    # Force rerun to update UI
                    st.rerun()
        
        # Loaded documents
        if st.session_state.documents_loaded:
            st.markdown("### Loaded Documents")
            # Extract unique document names
            doc_names = set()
            for file_id in st.session_state.documents_loaded:
                doc_name = file_id.rsplit('_', 1)[0]  # Remove size from ID
                doc_names.add(doc_name)
            
            for doc in sorted(doc_names):
                st.markdown(f"- {doc}")
        
        
        # Clear button
        if st.button("Clear All Documents", use_container_width=True):
            vector_store.clear()
            vector_store.save()
            st.session_state.documents_loaded = set()
            st.session_state.chat_history = []
            st.success("All documents cleared!")
            st.rerun()

with right_col:
    # Display stats
    col1, col2 = st.columns(2, border=True, vertical_alignment="center")
    
    with col1:
        st.markdown(
            f"<span style='color:#ff9900ff; font-weight:bold;'>{len(st.session_state.documents_loaded)}</span>&nbsp; Documents",
            unsafe_allow_html=True
        )
    with col2:
        chunks_count = len(vector_store.chunks) if vector_store else 0
        st.markdown(
            f"<span style='color:#ff9900ff; font-weight:bold;'>{chunks_count}</span>&nbsp; Chunks",
            unsafe_allow_html=True
        )

    # Display chat history

    # Display chat history
    with st.container(height=340):
        for message in st.session_state.chat_history:
            avatar = (
                "https://raw.githubusercontent.com/eshitakundu/NeuraSeek/refs/heads/main/src/static/user.png"
                if message["role"] == "user"
                else "https://raw.githubusercontent.com/eshitakundu/NeuraSeek/refs/heads/main/src/static/assistant.png"
            )

            # Handle normal messages
            if message.get("type") != "sources":
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])
            # Handle special "sources" messages with an expander
            else:
                with st.expander("View Sources"):
                    st.markdown(message["content"])


    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.documents_loaded:
            st.warning("Please upload a document first!")
        else:
            # Add user message to session
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Show spinner while processing
            with st.spinner("Processing your question..."):
                # Generate assistant response
                relevant_chunks = vector_store.similarity_search(prompt, k=4)
                response = rag_chain.generate_response(prompt, relevant_chunks)

                # Add assistant response
                st.session_state.chat_history.append({"role": "assistant", "content": response})

                if relevant_chunks:
                    sources_text = ""
                    for i, (chunk, metadata) in enumerate(relevant_chunks):
                        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                        sources_text += f"**Source {i+1}:** {metadata['source']}\n{preview}\n\n---\n"

                    # Add a special message type for sources
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "type": "sources",
                        "content": sources_text
                    })

            st.rerun()