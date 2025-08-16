import os
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))

from PIL import Image
import torch
import matplotlib.pyplot as plt

from lib.RAG_Biomistral.get_chunks import extract_fulltext_from_pdf, split_text_into_chunks
from lib.RAG_Biomistral.get_embeddings import get_TEXTembeddings, get_IMAGEembeddings
import pickle

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

from transformers import BitsAndBytesConfig
from transformers import MistralForCausalLM
from transformers import AutoTokenizer
from torch.cuda.amp import autocast
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from lib.RAG_Biomistral.rag_templates import prompt_general, prompt_RNAevolution
import yaml

class DummyEmbeddingFunction(Embeddings):
    # Minimal dummy Embedding class to satisfy LangChain's FAISS wrapper
    def embed_documents(self, texts):
        raise NotImplementedError("This is a dummy. You already have embeddings.")

    def embed_query(self, text):
        raise NotImplementedError("This is a dummy. You already have embeddings.")

class RAG_Biomistral():
    """
    A class that performs RNA-related Retrieval-Augmented Generation (RAG) using a BioMistral and medalpaca-7b LLM.
    """
    def __init__(self, embedding_function=None):
        # Clear GPU memory
        torch.cuda.empty_cache()
        # Load configuration file
        with open("../configs/rna_rag.yaml", 'r') as f:
            self.global_config = yaml.safe_load(f)
        # Use provided embedding function or a dummy fallback
        self.embedding_function = embedding_function or DummyEmbeddingFunction()
        # Print CUDA info for debugging
        print("CUDA is available", torch.cuda.is_available())
        print(f"CUDA version: {torch.version.cuda}")
        print(torch.cuda.get_device_name())

        # Select the appropriate text/image embedding model based on config
        text_visual_encoder_choice = self.global_config["model_for_embeddings"]["text_visual_encoder_choice"]

        if text_visual_encoder_choice==0: # max phrase length only 64 tokens, too short
            self.models_dict = {"modelTXT_name": "google/siglip2-base-patch32-256",
                                "modelIMAGE_name": "google/siglip2-base-patch32-256"} #"google/siglip-so400m-patch14-384"
        elif text_visual_encoder_choice==1: # max phraselength only 77 tokens, too short
            self.models_dict = {"modelTXT_name": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", 
                        "modelIMAGE_name": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"}
        elif text_visual_encoder_choice==2:
            self.models_dict = {"modelTXT_name": "sentence-transformers/clip-ViT-B-32", 
                        "modelIMAGE_name": "sentence-transformers/clip-ViT-B-32"}
        elif text_visual_encoder_choice==3:
            self.models_dict = {"model_name": 'ViT-H-14', 
                        "pretrained": 'laion2b_s32b_b79k'}
        elif text_visual_encoder_choice==4:
            self.models_dict = {"model_name": "EVA02-E-14-plus",
                                "pretrained": "laion2b_s9b_b144k"}
        elif text_visual_encoder_choice==5:
            self.models_dict = {"model_name":"hf-hub:imageomics/bioclip", 
                                "pretrained":"laion2b_e32b_b82k"}
            
    def calc_book_embeddings(self, path2rnaBooks, path2bookEmbed, save=True):
        # Split the RNA book PDFs into token chunks for embedding
        # Set number of tokens in each chunk
        max_tokens = self.global_config["rag"]["book_tokens_per_page"]
        overlap=int(max_tokens * self.global_config["rag"]["chunk_overlap_ratio"])
        
        self.chunks = []
        print("Extracting book...")
        full_text = ""
        for i,book in enumerate(path2rnaBooks):
            full_text += extract_fulltext_from_pdf(book)
            chunks_book = split_text_into_chunks(full_text, max_tokens=max_tokens, overlap=overlap)
            print("Book {} contains {} pages".format(i+1, len(chunks_book)))
            self.chunks += chunks_book
        print("------------------------------")
        print(len(self.chunks), "chunks with", len(self.chunks[0].split()), "words each")
        # Save or load precomputed embeddings
        if save:
            self.book_embeddings = get_TEXTembeddings(self.chunks, self.models_dict)
            with open(path2bookEmbed, 'wb') as f:
                pickle.dump(self.book_embeddings, f)
        else:
            with open(path2bookEmbed, 'rb') as f:
                self.book_embeddings = pickle.load(f)
        

    def text2image_test(self, image_path, descriptions):
        # Display image and calculate similarity between image and text descriptions
        print("The model:", self.models_dict)

        # Show image
        print("Image from: https://commons.wikimedia.org/w/index.php?curid=23636924")
        image = Image.open(image_path).convert('RGB')
        plt.figure(figsize=(image.width / 400, image.height / 400))  # 4x smaller
        plt.imshow(image, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()
        # Calculate embeddings
        text_embeddings = get_TEXTembeddings(descriptions, self.models_dict)
        images = [image]
        image_embedding = get_IMAGEembeddings(images, self.models_dict)
        # Compare
        print("text_embeddings:", text_embeddings.shape)
        print("image_embedding:", image_embedding.shape)
        cos_sim = torch.nn.functional.cosine_similarity(text_embeddings, image_embedding)
        for i in range(len(text_embeddings)):
            print("Score {} for {}".format(f"{cos_sim[i].item():.3f}", descriptions[i]))

    def build_vector_storeFAISS(self):
        # Convert list to tensor if needed
        if isinstance(self.book_embeddings, list):
            self.book_embeddings = torch.stack(self.book_embeddings)
        # Ensure it's float32 and on CPU
        self.book_embeddings = self.book_embeddings.to(torch.float32).cpu()
        # Convert to numpy (still on CPU)
        #embeddings = book_embeddings.numpy()
        # Make sure embeddings are in the correct format: list of np.arrays or 2D np.array
        if isinstance(self.book_embeddings, list):
            book_embeddings_np = np.stack(self.book_embeddings)
        else:
            book_embeddings_np = self.book_embeddings  # Already a numpy array
        # Move to CPU + NumPy
        book_embeddings_np = [emb.cpu().numpy() for emb in self.book_embeddings]
        text_embedding_pairs = list(zip(self.chunks, book_embeddings_np))

        # LangChain builds the index on CPU
        index = FAISS.from_embeddings(text_embedding_pairs, self.embedding_function)

        return index, book_embeddings_np, "model"
    

    def load_llm(self):
        # Load the BioMistral or MedAlpaca LLM using bitsandbytes quantization
        if self.global_config["llm_model"]["inference_8_bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16, # torch.bfloat16
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True  # this allows partial CPU fallback
            )
            

        if self.global_config["llm_model"]["llm_model_choice"]==0:
            # first run takes 20 min to download the model in C:\Users\User\.cache\huggingface\hub\models--BioMistral--BioMistral-7B-TIES
            model_name = "BioMistral/BioMistral-7B-TIES"
            # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = MistralForCausalLM.from_pretrained(model_name, 
                                                    #torch_dtype=torch.float16,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto", 
                                                    #use_flash_attention_2=True,  # if supported
                                                    quantization_config=bnb_config,
                                                    )
            self.llm_model = torch.compile(self.llm_model)

        else:
            model_name = "medalpaca/medalpaca-7b" 
            # model_name = "medalpaca/medalpaca-13b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=bnb_config
            )
        print(self.llm_model.device)

    def preprocess(self): 
        # Preprocessing: Build the vector store and load the LLM

        # Display an image and show PDBprompt and sequence
        # print(self.main_sequence)
        # plt.figure(figsize=(self.pdbImages[0].width / 300, self.pdbImages[0].height / 300))  # 3x smaller
        # plt.axis('off')  # Hide axis for a cleaner view
        # plt.imshow(self.pdbImages[0])
        # plt.show()
        # print(self.PDBprompt)

        print("Building vector store...")
        self.index, self.embeddings, self.embedder = self.build_vector_storeFAISS()
        self.vectorstore = self.index

        print("Loading LLM...")
        self.load_llm()

    def prompt_RNAevolution(self, main_sequence, backfolded_rna_tuple, PDBprompt):
        # Build prompt for RNA ancestral sequence reconstruction
        # Both have {PDBprompt['general_info']["length"]} nucleotides length.
        # Propose a plausible ancestral RNA sequence, relying on both the current and reconstructed RNA sequences (each 100 nucleotides long).
        hairpins_text = ""
        if 'hairpins' in PDBprompt['general_info']:
            hairpins_list = PDBprompt['general_info']['hairpins']
            if hairpins_list:
                hairpins_text = "\n\n**Hairpins of the current RNA sequence:**\n"
                hairpins_text += ", ".join(PDBprompt['general_info']['hairpins'])
        length_main_sequence=PDBprompt['general_info']["length"]

        prompt = prompt_RNAevolution.format(length_main_sequence=length_main_sequence,
                                             main_sequence = main_sequence,
                                             hairpins_description = hairpins_text,
                                             reconstructed_sequence = backfolded_rna_tuple[0],
                                             reconstruction_score = round(float(backfolded_rna_tuple[1])*100, 1),
                                             length_min=length_main_sequence-10,
                                             length_max=length_main_sequence-1)
        return prompt
    

    def prompt_general(self, main_sequence, PDBprompt, rag_images):
        # Build a general functional prediction prompt using context from text and images
        filtered_helices_info = {k: v for k, v in PDBprompt['helices_info'].items() if k != 'full_RNA_sequence'}
        filtered_helices_info = json.dumps(filtered_helices_info, indent=4)

        question = "What is the likely function of this RNA sequence?"
        question = "What are the two most likely functions of this RNA sequence? Provide both a primary and a secondary hypothesis, with brief reasoning for each."

        combined_context = ""

        # === Question CONTEXT RETRIEVAL ===
        question_tensor = get_TEXTembeddings([question], self.models_dict)
        question_context_docs = self.vectorstore.similarity_search_by_vector(
            question_tensor[0].detach().cpu().numpy(), k=self.global_config["rag"]["num_question_context_pages"]
        )
        context_question = "\n".join([doc.page_content for doc in question_context_docs])
        combined_context += "Context for question: " + context_question

        # === IMAGE CONTEXT RETRIEVAL ===
        context_images = ""
        image_tensors = get_IMAGEembeddings(rag_images, self.models_dict)
        #if not isinstance(image_features, torch.Tensor):
        #    image_features = torch.tensor(image_features).to('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(image_tensors.shape[0]):
            # Perform similarity search for each image
            image_vector = image_tensors[i].detach().cpu().numpy()
            image_context_docs = self.vectorstore.similarity_search_by_vector(image_vector, 
                                                                         k=self.global_config["rag"]["num_image_context_pages"])
            # Join retrieved captions or text
            context_images += "\n".join([doc.page_content for doc in image_context_docs])
        combined_context += "Context for images: " + context_images   

        N_hairpins = len(PDBprompt['general_info']['hairpins'])
        helix_descriptions = ""
        for helix_name, helix_data in PDBprompt['helices_info'].items():
            if helix_name.startswith('helix'):
                helix_descriptions += f"- **{helix_name}** consists of **{helix_data['base_pairs']} base pairs**\n" 

        N_base_stacking = len(PDBprompt["general_info"]['stacks'])
        length_main_sequence=PDBprompt['general_info']["length"]

        prompt = prompt_general.format(combined_context=combined_context,
                                             length_main_sequence=length_main_sequence,
                                             main_sequence = main_sequence,
                                             N_hairpins = N_hairpins,
                                             helix_descriptions=helix_descriptions,
                                             N_base_stacking=N_base_stacking,
                                             question=question,
                                             )
        return prompt

    def generate_response(self, main_sequence, backfolded_rna_tuple, PDBprompt, 
                           rag_images):
        if self.global_config["llm_model"]["prompt_choice"]==0:
            prompt = self.prompt_general(main_sequence, PDBprompt, rag_images)
        elif self.global_config["llm_model"]["prompt_choice"]==1:
            prompt = self.prompt_RNAevolution(main_sequence, backfolded_rna_tuple, PDBprompt)
        elif self.global_config["llm_model"]["prompt_choice"]==2:
            # custom DEBUG prompts 
            prompt = """
                You are an expert in origins of life and RNA structural evolution.

                Question: What is RNA hairpin? Please describe its structure and functions opf RNA hairpin.

                Answer:
            """
            question = "What are main 5 different functions of RNA hairpin?"
            context = "An RNA hairpin is a common secondary structure formed when a single strand of RNA folds back."
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: "

        print("Please wait for LLM answer ...")

        # === GENERATE RESPONSE ===
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}

        with torch.inference_mode(), autocast():
            outputs = self.llm_model.generate(**inputs, max_new_tokens=300)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def process(self, main_sequence, backfolded_rna_tuple, rag_images, PDBprompt):
        response = self.generate_response(main_sequence=main_sequence,
                                           backfolded_rna_tuple=backfolded_rna_tuple,
                                           PDBprompt=PDBprompt, 
                                           rag_images=rag_images, 
                                            )
        print(response)
        return response




