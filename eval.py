#%matplotlib inline
from word2gm_loader import Word2GM
from quantitative_eval import *

text8_model_dir = 'modelfiles/test'
w2gm_text8_1s = Word2GM(text8_model_dir)

w2gm_text8_1s.visualize_embeddings()

w2gm_text8_1s.show_nearest_neighbors('brother', 0)
w2gm_text8_1s.show_nearest_neighbors('brother', 1)

quantitative_eval(model_names=[('text8', text8_model_dir)])

quantitative_scws_df(text8_model_dir)

calculate_entailment(text8_model_dir)

quantitative_eval_over_time(text8_model_dir)

quanteval_plot_ind(text8_model_dir)