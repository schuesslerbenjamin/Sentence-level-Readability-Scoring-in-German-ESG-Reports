conda create --name ARA-RF python=3.12 
conda activate ARA-RF
pip install -r 'requirements/ARA-RF.txt'


conda create --name ARA-LLM python=3.12 
conda activate ARA-LLM
pip install -r 'requirements/ARA-LLM.txt'

conda create --name ARA-Syntax python=3.12 
conda activate ARA-Syntax
pip install -r 'requirements/ARA-Syntax.txt'

conda create --name ARA-Transformer python=3.12 
conda activate ARA-Transformer
pip install -r 'requirements/ARA-Transformer.txt'

conda create --name ARA-MoE python=3.12 
conda activate ARA-MoE
pip install -r 'requirements/ARA-MoE.txt'