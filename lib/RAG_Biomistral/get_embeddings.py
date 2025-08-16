# Get embeddings for text and images
# Models are according to text_visual_encoder_choice parameter in config: configs/rna_rag.yaml
# Some models are on gpu, some on cpu

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel
import open_clip
import torch

def get_TEXTembeddings(text, models_dict):
    if 'modelTXT_name' in models_dict.keys():
        modelTXT_name = models_dict['modelTXT_name']
        if "sentence-transformers" in modelTXT_name:
            print("sentence-transformers")
            modelTXT = SentenceTransformer(modelTXT_name)
            with torch.no_grad():
                text_embeddings = modelTXT.encode(sentences=text, 
                                                show_progress_bar=True,
                                                convert_to_tensor=True).to("cuda")
        elif "siglip" in modelTXT_name:
            print("siglip")
            modelTXT = AutoModel.from_pretrained(modelTXT_name).eval().cuda()
            processor = AutoProcessor.from_pretrained(modelTXT_name)
            text_input = processor(text=text, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                text_embeddings = modelTXT.get_text_features(**text_input)      
        else:
            print("CLIP")
            modelTXT = CLIPModel.from_pretrained(modelTXT_name).eval().cuda()
            processor = CLIPProcessor.from_pretrained(modelTXT_name)
            inputs_text = processor(text=text, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                text_embeddings = modelTXT.get_text_features(**inputs_text)
    else:
        # print('pretrained')
        model_name = models_dict['model_name']
        pretrained = models_dict['pretrained']

        model, preprocess, tokenizer = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        #model = model.eval().cuda()
        model = model.eval().cpu()
        tokenizer = open_clip.get_tokenizer(model_name)
        #text_input = tokenizer(text).cuda()
        text_input = tokenizer(text).cpu()
        # Embeddings
        with torch.no_grad():
            text_embeddings = model.encode_text(text_input)
            
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    return text_embeddings

def get_IMAGEembeddings(images, models_dict):
    if 'modelIMAGE_name' in models_dict.keys():
        modelIMAGE_name = models_dict["modelIMAGE_name"]
        if "sentence-transformers" in modelIMAGE_name:
            print("sentence-transformers")
            modelTXT = SentenceTransformer(modelIMAGE_name)
            with torch.no_grad():
                image_embedding = modelTXT.encode(sentences=images, 
                                                show_progress_bar=True,
                                                convert_to_tensor=True).to("cuda")
        elif "siglip" in modelIMAGE_name:
            print("siglip")
            modelIMAGE = AutoModel.from_pretrained(modelIMAGE_name).eval().cuda()
            processor = AutoProcessor.from_pretrained(modelIMAGE_name)
            # text_input = processor(text=text[0], return_tensors="pt").to("cuda")
            inputs = processor(images=images, return_tensors="pt").to("cuda")
            with torch.no_grad():
                image_embedding = modelIMAGE.get_image_features(**inputs)
        elif ("CLIP" in modelIMAGE_name) or ("clip" in modelIMAGE_name):
            print("CLIP")
            modelIMAGE = CLIPModel.from_pretrained(modelIMAGE_name).eval().cuda()
            processor = CLIPProcessor.from_pretrained(modelIMAGE_name)
            inputs = processor(text=[""], images=images, 
                               return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                outputs = modelIMAGE(**inputs)
                image_embedding = outputs.image_embeds  # Aligned image embedding
    else:
        #print('pretrained')
        model_name = models_dict['model_name']
        pretrained = models_dict['pretrained']

        model, preprocess, tokenizer = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        #model = model.eval().cuda()
        model = model.eval().cpu()
        #images = [image, image]
        #image_input = torch.stack([preprocess(img) for img in images]).cuda()
        image_input = torch.stack([preprocess(img) for img in images]).cpu()
        #image_input = preprocess(image_input).unsqueeze(0).cuda()
        with torch.no_grad():
            image_embedding = model.encode_image(image_input)
    image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)
    return image_embedding