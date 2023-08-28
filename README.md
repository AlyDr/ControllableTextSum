# ControllableTextSum

This repository contains a code for the approach to controllable text summarization described in the article "On the Way to Controllable Text Summarization".
The code is an adaptation of the HydraSum architecture for the Russian language. The original code alongside the methodology was presented by Tanya Goyal, Nazneen Fatema Rajani, Wenhao Liu, Wojciech Kryściński in their article "HydraSum - Disentangling Stylistic Features in Text Summarization using Multi-Decoder Models". 

# Running the code
The first step before training the model is the dataset preprocessing. It is necessary for HydraSum architecture to add two additional columns to the summarization dataset (we use "Gazeta" dataset via HuggingFace API). To do so `gate_probs.py` file should be run.

To train the model under the unguided setting the following command should be run:
```
!python3 train_seq2seq.py \
    --model_type mbart_mult_heads_2 \
    --model_name_or_path facebook/mbart-large-cc25 \
    --do_train \
    --train_data_file #TRAINDATA \
    --eval_data_file #EVALDATA \
    --test_data_file #TESTDATA \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --output_dir #OUTPUTDIRECTORY \
    --num_decoder_layers_shared 8
  ```

Where `#TRAINDATA`, `#EVALDATA` and `#TESTDATA` are the directories with train test and validation data and `#OUTPUTDIRECTORY` is where the model is saved. To train under the guided setting, additionally set the `use_sentence_gate_supervision`.



