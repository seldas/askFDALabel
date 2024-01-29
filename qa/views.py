from django.shortcuts import render
from django.http import JsonResponse
import html
import os, re

from xml.etree import ElementTree as ET

from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings.sentence_transformer import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

import torch
from peft import PeftModel
import transformers
import oracledb

############################## Support functions ############################################
def load_db(db_name = 'NDA_all-mpnet-base-v2-meta', embeddings='sentence-transformers/all-mpnet-base-v2'):
    db_folder = '/main/Docker/LabelGPT/LabelGPT/embeddings/'
    pd = db_folder + db_name
    embeddings = HuggingFaceEmbeddings(cache_folder='/main/Docker/LabelGPT/LabelGPT/models', 
                                       model_name=embeddings)
    
    db = Chroma(persist_directory=pd, embedding_function=embeddings)

    return db

def load_qamodel(qa = 'gpt4-x-alpaca', task = 'text-generation', model_max_length=1024):
    if qa == 'Platypus-13b':
        model_path = 'Open-Orca/OpenOrca-Platypus2-13B'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    elif qa == 'Llama2-13b':
        model_path = 'meta-llama/Llama-2-13b-chat-hf'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    elif qa == 'falcon-40b':
        model_path = 'tiiuae/falcon-40b-instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    
    model.config.max_new_tokens = 512
    model = model.eval()
    
    return tokenizer, model  

def prompt_prep(query, input_reference):
    input_text = ('Answer the question: ['+
                       query+
                      '], only based on the Input content: \n'+
                      '[ '+input_reference+']\n'+
                      '### Response: \n'
                     )
    return input_text

def prompt_prep_llama2(query, input_reference):
    input_text = ('<s>[INST] <<SYS>>Answer the question only based on the Input content.<</SYS>>\n"'+
                      'Question: '+query+'\n'+
                      'Input: '+input_reference+'\n'+
                       '<<SYS>> Summarize the result if possible. No more than 100 words.<</SYS>>\n'+
                      '[/INST]'
                     )
    return input_text.strip()

def prompt_prep_llama2_sequential(query, prev_answer='', input_reference=''):
    input_text = ('<s>[INST] <<SYS>>refine the previous answer based on the Input content. Provide concise answer.<</SYS>>\n"'+
                      'Question: '+query+'\n'+
                      'Previous Answer:'+prev_answer+'\n'+
                      'Input: '+input_reference+'\n'+
                      #'<<SYS>> Summarize the result if possible. No more than 100 words.<</SYS>>\n'+
                      '[/INST]'
                     )
    return input_text.strip()
###############################################################################################

    
################## Pre-load models and db when the server starts #########################
DEVICE = 'cuda:7' # device to handle input sequence; using the last GPU which usually has less usage.

embeddings = HuggingFaceEmbeddings(cache_folder='/main/Docker/LabelGPT/LabelGPT/models', 
                                       model_name='sentence-transformers/all-mpnet-base-v2')
db = FAISS.load_local("/main/Docker/LabelGPT/LabelGPT/embeddings/FDALabel_AE-chunks-mpnet-base-v2-faiss", embeddings)
tokenizer, model = load_qamodel(qa='Llama2-13b') # the llama2-13b-chat model was used for demo; can be changed.

pad_token_id = tokenizer.eos_token_id
eos_token_id = tokenizer.eos_token_id

# Configurations for the generative model, by default
generation_config = GenerationConfig(
                temperature=0.1,
                top_p=0.75,
                repetition_penalty=1.1,
            )
##########################################################################################


# in current version we only focus on the following labeling sections, for a more efficient demo
section_dict = {'34066-1':'BOXED WARNING SECTION', 
                '34067-9':'INDICATIONS & USAGE SECTION', 
                '59845-8':'INSTRUCTIONS FOR USE SECTION', 
                '34070-3':'CONTRAINDICATIONS SECTION', 
                '43685-7':'WARNINGS AND PRECAUTIONS SECTION', 
                '34071-1':'WARNINGS SECTION', 
                '42232-9':'PRECAUTIONS SECTION', 
                '34084-4':'ADVERSE REACTIONS SECTION', 
                '34073-7':'DRUG INTERACTIONS SECTION',
               }

def home(request):
    return render(request, 'qa.html', )

# customizes for labeling document analysis
def response(request): 
    if request.method == 'POST':
        # Get the variable from the POST data
        query = request.POST.get('message')
        keyword = re.search(r'{(.+)}',query)
        if keyword:
            prev_flag, candidates = accurate_search_mode(query, keyword[1], k=2)
        else:
            prev_flag = ''
            candidates = db.similarity_search(query, k=5)
        history = request.POST.get('history')
        
        response = []
        history_context = "Previous questions and responses:\n"+history+"\n"
        # print(query, candidates)
        for ind, c in enumerate(candidates):
            input_text = prompt_prep_llama2(query, c.page_content)
              
            context = history_context+input_text
            if len(context)<4000:
                # if the candidates is short, using the whole content in one LLM;
                inputs = tokenizer.encode(context, return_tensors='pt').to(DEVICE)
                
                with torch.inference_mode():
                    outputs = model.generate(
                            input_ids=inputs,
                            generation_config=generation_config,
                            max_new_tokens=512,
                            early_stopping=True,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id
                    )
    
                    generated_text = tokenizer.decode(outputs[0])
                    answer = re.split(re.escape('[/INST]'),generated_text) # for Llama-2
                    # answer = re.split('### Response:',generated_text) # for Llama-2
                    # print(answer)
                    if len(answer) == 2:
                        final_answer = re.sub(r'<\S+>','', answer[1]).strip()
                    else:
                        final_answer = re.sub(r'<\S+>','', answer).strip()
            else:
                # else, split the whole context into several smaller pieces;
                whole_text = c.page_content
                n = (len(whole_text) // 3000) + 1
                prev_answer = ''
                for i in range(n):
                    curr_text = whole_text[max(0, i*3000-50): min((i+1)*3000+50, len(whole_text))]
                    input_text = prompt_prep_llama2_sequential(query, prev_answer, curr_text)
                    if i == 0:
                        input_text = history_context+input_text
                    inputs = tokenizer.encode(input_text, return_tensors='pt').to(DEVICE)
                    
                    with torch.inference_mode():
                        outputs = model.generate(
                                input_ids=inputs,
                                generation_config=generation_config,
                                max_new_tokens=256+i*64,
                                early_stopping=True,
                                pad_token_id=pad_token_id,
                                eos_token_id=eos_token_id
                        )
        
                        generated_text = tokenizer.decode(outputs[0])
                        answer = re.split(re.escape('[/INST]'),generated_text) # for Llama-2
                        # answer = re.split('### Response:',generated_text) # for Llama-2
                        # print(answer)
                        if len(answer) == 2:
                            prev_answer = re.sub(r'<\S+>','', answer[1]).strip()
                        else:
                            prev_answer = re.sub(r'<\S+>','', answer).strip()
                    # print(prev_answer)
                final_answer = prev_answer            
            
            # Process the answer and add it to the final output
            if final_answer:
                final_answer = re.sub(r'\n','<br>',html.escape(final_answer))
                used_ref = re.sub(r'\n','<br>',html.escape(c.page_content))
                section_name = section_dict[c.metadata['sec_id']]
                response.append([prev_flag+final_answer, c.metadata['set_id'], c.metadata['genrname'], c.metadata['sec_id'], used_ref,
                                 section_name])
                
        # print(response)
        # Return the result as a JSON response
        return JsonResponse({'result': response})
 
def accurate_search_mode(query, keyword, k=3):
    ## This is the internal FDALabel API server, please replace it with your own server if necessary.
    dsnStr = oracledb.makedsn(str(os.environ["FDALABEL_SERVER"]),str(os.environ["FDALABEL_SERV_PORT"]),str(os.environ["FDALABEL_SERV_NAME"]))
    con = oracledb.connect(user=str(os.environ["FDALABEL_USER"]), password=str(os.environ["FDALABEL_PSW"]), dsn=dsnStr)
    cursor=con.cursor()
    ## END ###
    
    results = get_related_sections(cursor, drug_name=keyword)
    if results:
        docs_all = []
        for set_id, prodname, genrname, loinc_code, loinc_name, title, content_XML in results:
            text = ET.tostring(ET.fromstring(content_XML), method='text').decode('iso8859-1')
            text = re.sub(r'\s+',' ', text).strip()
            # n = (len(text) // 3000) + 1
            doc =  Document(page_content=text, 
                            metadata={"set_id":set_id, "prodname":prodname, "genrname":genrname,
                                        "section_title":title, "section_name":loinc_name,
                                        "sec_id":loinc_code}) # , "piece":i+1, "total_piece":n
            docs_all.append(doc)
        
        db_specific = FAISS.from_documents(docs_all, embeddings)
        prev_flag = ''
        result = db_specific.similarity_search(query, k=k)
    else:
        prev_flag = '<b><font style="color:red">Warning: Cannot find the drug name, the result may be inaccurate!</font></b><br>'
        result = db.similarity_search(query, k=k)
        
    return prev_flag, result 
    
def get_related_sections(cursor, drug_name='Abacavir'):
    drug_name = drug_name.upper()
    query_exact = f"""
        select l.set_id, l.product_normd_generic_names
            from druglabel.sum_spl l
            join druglabel.spl_sec s on s.spl_id = l.spl_id
            where (Upper(l.PRODUCT_NAMES)='{drug_name}') or (Upper(l.product_normd_generic_names)='{drug_name}') 
            and l.NUM_ACT_INGRS=1
            and l.document_type_loinc_code in ('34391-3','45129-4')
            order by l.eff_time desc
    """
    
    query_rx = f"""
        select l.set_id, l.product_normd_generic_names
            from druglabel.sum_spl l
            where REGEXP_LIKE(l.PRODUCT_NAMES, '{drug_name}', 'i') or REGEXP_LIKE(l.product_normd_generic_names, '{drug_name}', 'i')
            and l.NUM_ACT_INGRS=1
            and l.document_type_loinc_code in ('34391-3','45129-4')
            order by l.eff_time desc
    """

    query_contain = f"""
        select l.set_id, l.product_normd_generic_names
            from druglabel.sum_spl l
            where REGEXP_LIKE(l.PRODUCT_NAMES, '{drug_name}', 'i') or REGEXP_LIKE(l.product_normd_generic_names, '{drug_name}', 'i')
            and l.NUM_ACT_INGRS=1
            order by l.eff_time desc
    """
    
    result = cursor.execute(query_exact).fetchall()
    if not result:
        result = cursor.execute(query_rx).fetchall()
    if not result:
        result = cursor.execute(query_contain).fetchall()
        
    if result:
        for item in result:
            setid = item[0]
            query_getsection = f"""
                select l.set_id, l.product_names, l.product_normd_generic_names, s.loinc_code, st.loinc_name, s.title, s.content_XML
                from druglabel.spl_sec s
                join druglabel.section_type st on s.loinc_code = st.loinc_code
                join druglabel.sum_spl l on l.spl_id = s.spl_id
                where l.set_id='{setid}'
                and s.loinc_code in ('34066-1', 
                                   '34067-9', '59845-8', 
                                   '34070-3', 
                                   '43685-7', '34071-1', '42232-9', 
                                   '34084-4', 
                                   '34073-7')
            """ # here we hard coded these section ids just for simplicity; can be modified later.
            section_result = cursor.execute(query_getsection).fetchall()

            if len(section_result):
                return section_result
    else:
        return None