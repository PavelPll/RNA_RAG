# RNA_transformer
RNA Sequence Analysis Using Transformer Models
## Description 
* Vanilla or/and BERT-like transformer implementation to generate RNA sequence embeddings 
* RNA embeddings arithmetic
* Peptidyl Transferase Center (PTC): structure and evolution
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

* Install ViennaRNA from [here](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html)
* Install StaVia from [here](https://pyvia.readthedocs.io/en/latest/Installation.html)
```
git clone https://github.com/PavelPll/RNA_transformer.git
cd RNA_transformer
```
```
conda create -n rna python=3.10.16
conda activate rna
pip install -r requirements.txt
```

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

