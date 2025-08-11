# RNA_RAG
Exploring the Use of Retrieval-Augmented Generation (RAG) in RNA Sequence Analysis

## Description 
* RAG context construction
    * From textbooks
    * From images
    * from RNA 3D structure
    * from RNA diffusion model
* Example: Using RAG to Decode and Evolve RNA Molecules
* For more information click [here](https://github.com/PavelPll/RNA_transformer/blob/main/docs/RNA_transformer.pdf)



## Getting Started

### Dependencies
* The starting point is [Vanilla Transformer implementation](https://github.com/hkproj/pytorch-transformer), used to translate phrases from English to Italian
* [RNAcentral DATABASE](https://rnacentral.org) of non-coding RNA (ncRNA) sequences
* [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) to get some physicochemical properties
* [RNA-FM](https://huggingface.co/multimolecule/rnafm): Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions
* [ViennaRNA](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html) predicting and comparing RNA secondary structures
* [StaVia (Via 2.0)](https://pyvia.readthedocs.io/en/latest/pyVia-home.html): single-cell trajectory inference method
* [IUPAC code](https://www.bioinformatics.org/sms/iupac.html) for nucleotides and amino acids
* Windows 11, Visual Studio Code
* Torch

### Installing

I adapted the same conda environment, rna_rag, for both RAG and Ribodiffusion. DRfold2, however, I installed in a Docker container running Ubuntu 22.04, due to the ARENA package requiring Linux for compilation. * Install DRfold2:
```
git clone https://github.com/PavelPll/RNA_RAG.git
cd RNA_RAG
git clone https://github.com/leeyang/DRfold2.git drfold2
git clone https://github.com/pylelab/Arena.git drfold2/Arena
cd drfold2
mkdir file_exchange\fasta_input && mkdir file_exchange\pdb_output
docker build -t drfold_image ../
docker run --gpus all -it --name drfold_container -v .:/opt/drfold2 drfold_image bash
Run inside container:
wget --header="User-Agent: Mozilla/5.0" https://zhanglab.comp.nus.edu.sg/DRfold2/res/model_hub.tar.gz
tar -xzvf model_hub.tar.gz
rm -rf model_hub.tar.gz
cd Arena
make Arena
# clang++ -O3 Arena.cpp -o Arena
exit
cd ..
```

* install Ribodiffusion:
```
git clone https://github.com/ml4bio/RiboDiffusion
cd RiboDiffusion
Model checkpoint can be downloaded from here. 
https://drive.google.com/drive/folders/10BNyCNjxGDJ4rEze9yPGPDXa73iu1skx
Another checkpoint trained on the full dataset (with extra 0.1 Gaussian noise for coordinates) can be downloaded from here.
https://drive.google.com/file/d/1-IfWkLa5asu4SeeZAQ09oWm4KlpBMPmq/view
Download and put the checkpoint files in the RiboDiffusion/ckpts folder.
```
```
conda create -n rna python=3.10.16
conda activate rna
pip install -r requirements.txt
```
* Install ViennaRNA from [here](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html)
* Install StaVia from [here](https://pyvia.readthedocs.io/en/latest/Installation.html)

### Executing program

* #### Extract/Generate RNA sequence data 
    * Extraction RNA sequences from RNAcentral database
         ```
         cd scripts && python rna_data_extract.py
         ```
    * Generate real RNA sequences of interest:
        ```
        cd scripts && python rna_data_extract_unique.py
        ```
    * Generate random/symmetric RNA sequences:
        ```
        cd scripts && python rna_data_extract_random.py
        cd scripts && python rna_data_extract_symmetric.
        ```
    * Generate RNA hairpins:
        ```
        cd scripts && python rna_data_extract_hairpin.py
        cd scripts && python rna_data_extract_hairpin_tail.py
        ```
* #### Run RNA_transformer
    * Define transformer parameters in config models/RNA_transformer/config.py
      
        * Construct a composite loss
            ```
            "autoencoder_bert": False,
            "autoencoder_vanilla": True,
            "classification": True,
            ```
        *  Set whether to use or not:
            * learnable loss weighting for the classification task
            * custom embeddings  
            * a mask token for training
        * etc
    * Run training and inference
        ```
        cd notebooks && RNA_transformer.ipynb
        ```
* #### Analysis
    * Calculate and plot RNA secondary structure:
        ```
        cd notebooks && RNA_plot.ipynb
        ```
    * Check embeddings arithmetic and the effect of basis:
        ```
        cd notebooks && embed_arithmetic_kdtree.ipynb
        cd notebooks && embed_arithmetic_basis.ipynb
        ```
    * Run Single-Cell Inspired Analysis of RNA
        ```
        cd notebooks && VIA_analysis.ipynb
        ```

## License
This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details



> [!NOTE]
> For more information see short [presentation](https://github.com/PavelPll/RNA_transformer/blob/main/docs/RNA_transformer.pdf)

