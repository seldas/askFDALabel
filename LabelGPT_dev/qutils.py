from langchain.embeddings.sentence_transformer import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
# from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer, LlamaForCausalLM


def load_db(embeddings='distilbert'):
    if embeddings == 'distilbert':
        persist_directory = "embeddings/NDA_qa_distilbert_cos_v1/"
        embedding_model = 'multi-qa-distilbert-cos-v1'
    elif embeddings == 'mpnet-v2':
        persist_directory = "embeddings/NDA_all-mpnet-base-v2/"
        embedding_model = 'sentence-transformers/all-mpnet-base-v2'
    elif embeddings == 'glove':
        persist_directory = "embeddings/NDA_glove_840B/"
        embedding_model = 'average_word_embeddings_glove.840B.300d'

    embeddings = HuggingFaceEmbeddings(cache_folder='/main/Docker/LabelGPT/LabelGPT/models', 
                                   model_name=embedding_model)

    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def load_qamodel(qa = 'gpt4-x-alpaca', task = 'text-generation', model_max_length=1024):
    if qa == 'bloom':
        qa_model = "bigscience/bloom"
        tokenizer = AutoTokenizer.from_pretrained(qa_model, model_max_length = model_max_length, unk_token ="<s>")
        model = AutoModel.from_pretrained(qa_model)
    elif qa == 'flan-t5-large':
        qa_model = "google/flan-t5-large"
        tokenizer = T5Tokenizer.from_pretrained(qa_model, model_max_length = model_max_length, unk_token ="<s>")
        model = T5ForConditionalGeneration.from_pretrained(qa_model)
    elif qa == 'gpt4-x-alpaca':
        qa_model = "chavinlo/gpt4-x-alpaca"
        tokenizer = AutoTokenizer.from_pretrained(qa_model, model_max_length = model_max_length, unk_token ="<s>")
        model = LlamaForCausalLM.from_pretrained(qa_model, device_map="auto")
    elif qa == 'open_llama':
        qa_model = "openlm-research/open_llama_13b"
        tokenizer = AutoTokenizer.from_pretrained(qa_model, model_max_length = model_max_length, unk_token ="<s>")
        model = AutoModel.from_pretrained(qa_model, device_map="auto")
    elif qa == 'ChatGPT':
        os.environ['OPENAI_API_KEY']='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'
        llm = OpenAI()

    model.config.max_new_tokens = 512
    
    return tokenizer, model  