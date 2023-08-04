'''Get cloze values'''
import json
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import openai
import math
import numpy as np
import itertools

key_fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/lab_api_key.txt'
with open(key_fname,'r') as file:
    key = file.read()
openai.api_key = key #get it from your openai account


def get_completions(prompt):
    '''get the cloze values for every token in the input
    prompt: text input in a list'''
    completions = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=0,
        top_p=1,
        logprobs = 0,
        frequency_penalty=0,
        presence_penalty=0, 
        echo=True
    )    
    logprobs = completions["choices"][0]["logprobs"]["token_logprobs"]
    tokens = completions["choices"][0]["logprobs"]["tokens"]
    probs = [np.e ** prob if prob is not None else 1 for prob in logprobs]
    df = pd.DataFrame({'tokens':tokens,
                       'probs':probs})
    return df
    

def get_word_cloze(df,prompt):
    '''get the probability of every word in the input
    df: output of the model: 'tokens' and 'probs'
    prompt: list of text input used for calculating the token probability'''
    df['tokens'] = df.apply(lambda row: "" if 'bytes' in row['tokens'] else row['tokens'], axis=1)
    utterance = prompt.split()
    probs = []
    prob_list = []
    token_prob = 1
    placeholder = ""
    word_count = 0
    for i, token in enumerate(df["tokens"]):
        placeholder += token.strip()
        prob = df.loc[i, "probs"]
        if placeholder == utterance[word_count].strip().replace("’",""): #to deal with "’", e.g. couldn’t
            if prob_list:
                token_prob *= np.prod(prob_list)
            token_prob *= prob
            probs.append(token_prob)
            placeholder = ""
            word_count += 1
            prob_list = []
            token_prob = 1
        else:
            prob_list.append(prob)
    df_cloze = pd.DataFrame({'words':utterance,
                       'probs':probs})
    return df_cloze



def extract_cloze(prompt):
    df = get_completions(prompt)
    df_cloze = get_word_cloze(df,prompt)
    return df_cloze


def pickle_to_csv(cloze_df, cond_name):
    '''Get the data frame series to csv file'''     
    dfs_with_item_id = []
    for item_id, df in enumerate(cloze_df):
        df_with_item_id = df.copy()
        df_with_item_id['item ID'] = item_id + 1 
        dfs_with_item_id.append(df_with_item_id)
    combined_df = pd.concat(dfs_with_item_id)
    combined_df['Word Position'] = combined_df.index + 1
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df[['item ID', 'Word Position', 'words', 'probs']]
    fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/cloze/all_' + cond_name + '.csv'
    combined_df.to_csv(fname, index=False)


def cloze_last_word(cloze_df):
    '''Get the cloze values of the CWs of all conditions together'''
    dfs_last_word = []
    for item_id, df in enumerate(cloze_df):
        df_last_word = df.tail(1).copy() 
        df_last_word.loc[:, 'item ID'] = item_id + 1 
        dfs_last_word.append(df_last_word)
    combined_df = pd.concat(dfs_last_word)
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df[['item ID', 'words', 'probs']]
    return combined_df


def plot_words_cloze(df):    
    '''plot cloze values for each word in a sentence
    df: every word and its associated cloze value'''
    data = df['probs']
    labels = df['words']
    tick_positions = np.arange(len(data))
    ax = plt.figure(figsize=(10, 8)).add_subplot(111)
    ax.plot(tick_positions, data)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, rotation=45)
    plt.tight_layout()
    plt.show()
    
    
## --------------------------------------------
# read data
## --------------------------------------------
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/stimuli/InfoPos_Stim_Final_with_Lexical_Characteristics.xlsx'
df = pd.read_excel(fname)
df_sel = df.loc[:183,['info','noinfo','expected','unexpected','LC_pre','HC_uninfo_pre']]

LC_info = df_sel['LC_pre'] + ' ' + df_sel['info']
LC_uninfo = df_sel['LC_pre'] + ' ' + df_sel['noinfo']
HC_exp = df_sel['HC_uninfo_pre'] + ' ' + df_sel['expected']
HC_unexp = df_sel['HC_uninfo_pre'] + ' ' + df_sel['unexpected']

## --------------------------------------------
# get cloze
## --------------------------------------------
cloze_LC_info = LC_info.apply(extract_cloze)
cloze_LC_uninfo = LC_uninfo.apply(extract_cloze)
cloze_HC_exp = HC_exp.apply(extract_cloze)
cloze_HC_unexp = HC_unexp.apply(extract_cloze)

# Save the DataFrame
pickle_fname=r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/cloze/'
with open(pickle_fname + 'LC_info.pickle', 'wb') as file:
    pickle.dump(cloze_LC_info, file)

with open(pickle_fname + 'LC_uninfo.pickle', 'wb') as file:
    pickle.dump(cloze_LC_uninfo, file)

with open(pickle_fname + 'HC_exp.pickle', 'wb') as file:
    pickle.dump(cloze_HC_exp, file)

with open(pickle_fname + 'HC_unexp.pickle', 'wb') as file:
    pickle.dump(cloze_HC_unexp, file)



## -----------------------------------------------------------
# Read data containing cloze values for each sentence and plot
## -----------------------------------------------------------
# Load the DataFrame
pickle_fname=r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/cloze/'
with open(pickle_fname + 'LC_info.pickle', 'rb') as file:
    cloze_LC_info = pickle.load(file)
    
with open(pickle_fname + 'LC_uninfo.pickle', 'rb') as file:
    cloze_LC_uninfo = pickle.load(file)
    
with open(pickle_fname + 'HC_exp.pickle', 'rb') as file:
    cloze_HC_exp = pickle.load(file)
    
with open(pickle_fname + 'HC_unexp.pickle', 'rb') as file:
    cloze_HC_unexp = pickle.load(file)

# Save the cloze values of all words in a csv file
pickle_to_csv(cloze_LC_info, 'LC_info')
pickle_to_csv(cloze_LC_uninfo, 'LC_uninfo')
pickle_to_csv(cloze_HC_exp, 'HC_exp')
pickle_to_csv(cloze_HC_unexp, 'HC_unexp')

# Save the cloze values of only the CWs of all conditions
combined_LC_info = cloze_last_word(cloze_LC_info)
combined_LC_uninfo = cloze_last_word(cloze_LC_uninfo)
combined_HC_exp = cloze_last_word(cloze_HC_exp)
combined_HC_unexp = cloze_last_word(cloze_HC_unexp)
combined_df = pd.concat([combined_LC_info, combined_LC_uninfo[['words','probs']],
                        combined_HC_exp[['words','probs']], combined_HC_unexp[['words','probs']]], axis=1)
new_column_names = {
    'item ID': 'item ID',
    'words': 'LC_info_words',
    'probs': 'LC_info_probs',
    'words.1': 'LC_uninfo_words',
    'probs.1': 'LC_uninfo_probs',
    'words.2': 'HC_exp_words',
    'probs.2': 'HC_exp_probs',
    'words.3': 'HC_unexp_words',
    'probs.3': 'HC_unexp_probs',
}
combined_df.rename(columns=new_column_names, inplace=True)
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/InfoPos/cloze/allConds_CWs.csv'
combined_df.to_csv(fname, index=False)


# plot the cloze values for each item
isen = 5
plot_words_cloze(cloze_LC_info.iloc[isen])
plot_words_cloze(cloze_LC_uninfo.iloc[isen])
plot_words_cloze(cloze_HC_exp.iloc[isen])
plot_words_cloze(cloze_HC_unexp.iloc[isen])
