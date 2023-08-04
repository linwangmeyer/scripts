import json
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import re

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

######################################################################
####### Load Infopos stimuli #######
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/stimuli/InfoTopic_stim_toyData_gpt4_output.csv'
df = pd.read_csv(fname)

# ----------------------------------------------------------------
# Get the pairwise similarity of the generated text
# ----------------------------------------------------------------

def get_sentence_similarity(df):
    '''Get the pairwise similarity for all stimuli of one item
    df: dataframe series containing text stimuli'''  
    df.tolist()
    sen_emb = model.encode(df) #Nparagraph x vectorDimension
    r = cosine_similarity(sen_emb)
    indices = np.triu_indices(r.shape[0],k=1)
    r_vec = r[indices]
    return r_vec

Rs = []
for index, row in df.iterrows():
    r_vec = get_sentence_similarity(row.filter(like='gpt4'))
    Rs.append(r_vec)

# average pairwise similarity for each item
result = np.mean(np.array(Rs),axis=1)
df['mean_similarity'] = result

# plot the similarity values for each item
plt.figure(figsize=(10,6))
plt.bar(df.index, df['mean_similarity'])
plt.xlabel('condition')
plt.ylabel('mean similarity')
plt.title('mean simialrity value by condition')
plt.xticks(df.index, df['constraint'], rotation=45)
plt.tight_layout()
plt.savefig(r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/plots/SenSim_generated_30.png')
plt.show()



# -----------------------------------------------------------------------
# Get the sentence-level similarity before and after CW onset up to CW
# -----------------------------------------------------------------------
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/ALL_EXPERIMENTAL_STIMULI/InfoTopic_stim_toyData_gpt4_output.csv'
df = pd.read_csv(fname)
df_sel = df[(df['constraint'] != 'HC') & (df['constraint'] != 'LC')][['constraint','context']]
df_sel.reset_index(drop=True, inplace=True)

def get_pre_post_similarity(df):
    '''Get the distance of the sentence embeddings before and after CW.
    df: dataframe series containing text stimuli including the CW'''  
    df_post = df_sel['context'].tolist()
    df_pre = df_sel['context'].str.rsplit(n=1).str[0].tolist()
    emb_pre = model.encode(df_pre) #Nparagraph x vectorDimension
    emb_post = model.encode(df_post) #Nparagraph x vectorDimension
    r = cosine_similarity(emb_pre, emb_post)
    sim = np.diag(r)
    return sim

sim = get_pre_post_similarity(df_sel['context'])
df_sel['pre_post_similarity'] = sim

# plot the similarity values for each item
plt.figure(figsize=(10,6))
plt.bar(df_sel.index, df_sel['pre_post_similarity'])
plt.xlabel('condition')
plt.ylabel('semantic similarity: pre vs post')
plt.title('similarity value by condition')
plt.xticks(df_sel.index, df_sel['constraint'], rotation=45)
plt.tight_layout()
plt.savefig(r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/plots/SenSimPrePost_30.png')
plt.show()

# ------------------------------------------------------------------------------
# Get the sentence-level similarity before and after CW onset: moving window
# Window size: defined based on the punctuation of the generated sentences
# ------------------------------------------------------------------------------
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/ALL_EXPERIMENTAL_STIMULI/InfoTopic_stim_toyData_gpt4_output.csv'
df = pd.read_csv(fname)
df_sel = df[(df['constraint'] != 'HC') & (df['constraint'] != 'LC')].filter(regex='constraint|context|gpt4')
df_sel.reset_index(drop=True, inplace=True)

def get_similarity_pre_post_wind(df_sel,post,nwind):
    '''Get pre-post similarity for defined window size
    window is defined based on the punctuations in the generated texts
    wind1 = cw only, wind2 = cw + first part of text, ...
    df_sel contains both context and generated texts
    post: name of generated text of one iteration, e.g. gpt4_p1
    nwind: number of windows
    Return: r_wind with a dimension of nitems x nwinds'''
    df_pre = df_sel['context'].str.rsplit(n=1).str[0]
    list_pre = df_pre.tolist() #the context without cw
    emb_pre = model.encode(list_pre) #Nparagraph x vectorDimension
    
    df_cw = df_sel['context'].str.rsplit(n=1).str[1]
    df_sel[post] = df_sel[post].apply(lambda x: " " + x if x and x[0].islower() else ". " + x if x and x[0].isupper() else x)
    df_post = df_cw + df_sel[post] #cw + generated texts
    df_post = df_post.str.replace('.', ',')
    len_text = df_post.str.split(',').apply(lambda x: len(x))
    if nwind > np.min(len_text):
        raise ValueError('the number of window exceeds the length of the generated texts')
    sim_winds = []
    for iwind in range(1,nwind+1):
        list_post = (df_pre + ' ' + df_post.str.split(',', n=iwind).str[:iwind].apply(lambda x: ','.join(x))).tolist()
        emb_post = model.encode(list_post) #Nparagraph x vectorDimension
        r = cosine_similarity(emb_pre, emb_post)
        sim_winds.append(np.diag(r))
    r_wind = np.concatenate(sim_winds).reshape(len(sim_winds),-1).transpose() #itemxwindow
    
    return r_wind, df_pre, df_post


sims = []
gpt4_columns = [col for col in df_sel.columns if 'gpt4' in col]
nwind=5
for column in gpt4_columns:
    sim,_,_ = get_similarity_pre_post_wind(df_sel,column,nwind)
    sims.append(sim)  
r = np.mean(np.stack(sims,-1),2) #nitem x nwind

colnames = ['wind' + str(i) for i in range(1, nwind+1)]
df_r = pd.DataFrame(r,columns=colnames)
df_check = pd.concat([df_sel,df_r],axis=1)

# plot the similarity values for each item, colored by size of window
df_r.plot(kind='bar', figsize=(10, 6))
plt.xlabel('condition')
plt.ylabel('mean semantic similarity: pre vs post')
plt.title('mean similarity value for each item')
plt.xticks(df_check.index, df_check['constraint'], rotation=45)
plt.tight_layout()
plt.legend(title='Legend', loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/plots/SenSimPrePost_wind5_30.png')
plt.show()

# Average across items
mean_values = df_check.groupby('constraint')[['wind1', 'wind2', 'wind3', 'wind4', 'wind5']].mean()
mean_values = mean_values.reset_index('constraint')
mean_values.plot(kind='bar', figsize=(10, 6))
plt.xlabel('condition')
plt.ylabel('semantic similarity: pre vs post')
plt.title('mean semantic similarity across 3 items')
plt.xticks(mean_values.index, mean_values['constraint'], rotation=45)
plt.tight_layout()
plt.savefig(r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/plots/SenSimPrePost_wind5_meanitems_30.png')
plt.show()

