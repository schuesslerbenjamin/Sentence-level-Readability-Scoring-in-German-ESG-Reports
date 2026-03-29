base_model_dir="models/"
ARA_Baseline_model_dir="${base_model_dir}ARA_Baseline/ARA_Baseline"
ARA_RF_model_dir="${base_model_dir}ARA_readability_formula_model/readability_formula_model"
ARA_LLM_model_dir="${base_model_dir}ARA_LLM_model/LLM_model"
ARA_Syntax_model_dir="${base_model_dir}ARA_Syntax_model/Syntax_model.json"
ARA_Transformer_model_dir="${base_model_dir}ARA_Transformer_model/Transformer_model"
ARA_MoE_model_dir="models/ARA_MOE_model/MOE_model"

# ARA task
echo "Evaluation"


echo -e "Sentence Length Baseline"
conda activate ARA-RF
python ARA_Baseline.py --model_path $ARA_Baseline_model_dir --mode eval

echo -e "\n\nReadability Formulae Baseline"
conda activate ARA-RF
python ARA_Readability_Formula_model.py --model_path $ARA_RF_model_dir --mode eval

echo -e "\n\nSyntactic Features Model"
conda activate ARA-Syntax
python ARA_Syntax_model.py --model_path $ARA_Syntax_model_dir --mode eval

echo -e "\n\nXLM RoBERTa base Model"
conda activate ARA-Transformer
python ARA_Transformer_model.py --model_path $ARA_Transformer_model_dir --mode eval

echo -e "\n\nLLM Model"
conda activate ARA-LLM
python ARA_LLM_model.py --model_path $ARA_LLM_model_dir --mode eval

echo -e "\n\nMean Combinations"
conda activate ARA-MoE
python ARA_combination_model.py --mode eval
