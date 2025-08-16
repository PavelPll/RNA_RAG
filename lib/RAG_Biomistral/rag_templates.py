# Only two kinds of prompts are possible: general and evolutionary. This is configured using the prompt_choice parameter 
# in configs/rna_rag.yaml. The general prompt (prompt_choice = 0) is used to ask a question about a given RNA, 
# while the evolutionary prompt (prompt_choice = 1) is used to generate a new RNA sequence. Both prompts can be 
# visualized and modified, if needed, in lib/RAG_Biomistral/rag_templates.py i.e. here. 

prompt_RNAevolution= """
    You are an expert in the origins of life and RNA structural evolution.

    **Task:** Propose a slightly shorter ancestral RNA sequence using both the current and reconstructed RNA sequences ({length_main_sequence} nucleotides each). 

    **Current RNA sequence:**  
    {main_sequence}{hairpins_description}

    **Reconstructed sequence from 3D structure of the current RNA sequence:**  
    {reconstructed_sequence}  
    (Sequence identity: {reconstruction_score}%)

    **Requirements for the proposed ancestral RNA sequence:**  
    - Nucleotide changes must not exceed approximately 1% of the total nucleotides in the ancestral RNA sequence, with respect to the current RNA sequence.
    - Length of the proposed ancestral RNA sequence must be between {length_min} and {length_max} nucleotides.  

    **Output format:**  
    Return the proposed ancestral RNA sequence in uppercase letters, no explanation. The sequence must be strictly shorter than the current RNA sequence.
    """


prompt_general= """
    You are an expert in the origins of life and RNA structural evolution.

    **Context:** {combined_context}

    **Task:** Answer the question based on the given non-coding RNA sequence.

    **[1] Non-coding RNA sequence:**  
    {main_sequence}

    **[2] Structural features of the non-coding RNA:**

    - {N_hairpins} hairpins  

    - {helix_descriptions}

    - {N_base_stacking} base-stacking interactions  

    **Question:**  
    {question}
    """