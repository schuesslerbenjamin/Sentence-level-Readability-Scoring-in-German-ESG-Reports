# Analyzing and Improving Readability in German Environmental, Social, and Governance Reporting

by Benjamin Josef Schüßler

## Getting Started

Make sure you have conda installed. Also make sure to add your huggingface token ([available here](https://huggingface.co/settings/tokens)) in the [local.env](local.env) file.

You need access to the following models on huggingface (You need to get permisison for the models from the repositories with *):
* https://huggingface.co/FacebookAI/xlm-roberta-base
* https://huggingface.co/FacebookAI/xlm-roberta-large
* https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct *
* https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct *
* https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct *
* https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 *
* https://huggingface.co/google/gemma-3-4b-it *
* https://huggingface.co/DEplain/trimmed_mbart_sents_apa_web




Also install the EASSE-DE repository as follows:

```
git clone https://github.com/rstodden/easse-de.git
cd easse-de
conda create --name easse-de python=3.12
conda activate easse-de
pip install -e .
pip install spacy_udpipe
pip install ipykernel

python -m spacy download de_dep_news_trf
```
In the file easse/sari.py, comment in the lines 253 - 256 and add:
```
sys_sents = [utils_prep.normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj) for sent in sys_sents]
refs_sents = [[utils_prep.normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj) for sent in ref_sents] for ref_sents in refs_sents]
```
on the same level as the line 256.
This was probably an oversight from the developers that adapted EASSE to German. This follows the original implementation of EASSE available [here](https://github.com/feralvam/easse/blob/master/easse/sari.py).

### Setting up the Environments for the Individual Models

The following installation steps can be abbreviated using the [installation script](installation_script.sh). The second command is required for the Google Gemma LLM. The third downloads the spacy model for the German language.
```
source installation_script.sh

apt-get install build-essential

python -m spacy download de_dep_news_trf
```


#### ARA Baseline and Readability Formula-based Model
```
conda create --name ARA-RF python=3.12 
conda activate ARA-RF
pip install -r 'requirements/ARA-RF.txt'
```

#### ARA LLM-based Model
```
conda create --name ARA-LLM python=3.12 
conda activate ARA-LLM
pip install -r 'requirements/ARA-LLM.txt'
```
For the Google Gemma LLM:
```
apt-get install build-essential
```

#### ARA Syntax-based Model
```
conda create --name ARA-Syntax python=3.12 
conda activate ARA-Syntax
pip install -r 'requirements/ARA-Syntax.txt'

python -m spacy download de_dep_news_trf
```

#### ARA Transformer-based Model
```
conda create --name ARA-Transformer python=3.12 
conda activate ARA-Transformer
pip install -r 'requirements/ARA-Transformer.txt'
```

#### ARA Combination Model
```
conda create --name ARA-MoE python=3.12 
conda activate ARA-MoE
pip install -r 'requirements/ARA-MoE.txt'
```

### Downloading the Models

The trained models are available at: https://huggingface.co/schuesslerbenjamin/ARA-TS-German-ESG.
Please download them into the [models](models) folder.


# Assessing the Readability

All ARA models offer the following modes: train, eval, cli

They can be specified by adding --mode [MODE] when executing the python file.

train: Trains the model with the hyperparameters set in the accompanying settings file. Also evaluates the model and saves the predictions so that the MoE model can be trained from them.

eval: Evaluates the model on the evaluation and evaluation experts data splits.

cli: Intended for use via the CLI. Rates the sentence provided via --input_sentence.

For example:
```
conda activate ARA-Syntax
python ./ARA_Syntax_model.py --mode eval
```


# Using the CLI Interface to Rate the Readability of Sentences
In a bash shell, run:

```
source cli.sh
```

The environments, the model, and the datasets must be correctly saved.
Please make sure you use the bash terminal or similar.