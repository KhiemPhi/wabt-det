from builtins import breakpoint
import re
import os 
from nltk.tokenize.punkt import PunktSentenceTokenizer
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
space_pattern = '\s+'
giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mention_regex = '@[\w\-]+'
emoji_regex = '&#[0-9]{4,6};'

import codecs
import glob


def clear_sem_eval_text(text):
    return text.strip().replace('\t', ' ').replace('\n', ' ')

def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = {}
    article_id_list, sentence_id_list, sentence_list = ([], [], [])
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with codecs.open(filename, "r", encoding="utf8") as f:
            articles[article_id] = f.read()
    return articles

def sents_token_bounds(text):
    sents_starts = []
    for start, end in PunktSentenceTokenizer().span_tokenize(text):
        sents_starts.append(start)
    sents_starts.append(100000)
    return np.array(sents_starts)

def get_context_sem_eval(article, span_start, span_end):
    bounds = sents_token_bounds(article)
    context_start = bounds[np.where(bounds <= span_start)[0][-1]]
    context_end = bounds[np.where(bounds >= span_end)[0][0]]
    return clear_sem_eval_text(article[context_start:context_end])

def load_data_sem_eval(args, file_path="sem-eval"):

    label_file = os.path.join(file_path, "dev-task-flc-tc.labels") 
    articles = read_articles_from_file_list('sem-eval/'+'dev-articles')
    #1. Filter to include whataboutism + non-whataboutism 
    articles_id, span_starts, span_ends, gold_labels = ([], [], [], [])
    with open(label_file, "r") as f:
        for row in f.readlines():
            article_id, gold_label, span_start, span_end = row.rstrip().split("\t")
            
            if "whataboutism" in gold_label.lower():
                numeric_label = 1 
            else: 
                numeric_label = 0 
            
            articles_id.append(article_id)
            gold_labels.append(numeric_label)
            span_starts.append(span_start)
            span_ends.append(span_end)

    #2.  Now let's build into dataframe
    data = pd.DataFrame.from_dict({'article_id': articles_id, 
              'article': [articles[id] for id in articles_id], 
              'span_start': np.array(span_starts).astype(int), 
              'span_end': np.array(span_ends).astype(int),
              'label': gold_labels
             })
    data['span'] = data.apply(lambda x: clear_sem_eval_text(x['article'][x['span_start']:x['span_end']]), axis=1)
    data['context'] = data.apply(lambda x: get_context_sem_eval(x['article'], x['span_start'], x['span_end']), axis=1) # this is what we need as sentences, now we need to mine for context


def get_data_loader(train_data, val_data, batch_size=5):
      
    # Create DataLoader for training data    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data    
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader


def scatter_tSNE(features, labels, file_path="vis/tSNE/test-SNE.jpg"):

    num_classes = 2 
    embeddings = TSNE(n_components=2).fit_transform(features)
   
    palette = np.array(sns.color_palette("hls", num_classes))
    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(embeddings[:,0], embeddings[:,1], lw=0, s=40, c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(embeddings[labels == i, :], axis=0)
        if i == 0:
            txt = ax.text(xtext, ytext, "NW", fontsize=24)
        else: 
            txt = ax.text(xtext, ytext, "W", fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig(file_path)
   

def plot_clustered_stacked(dfall, labels=None, title="Dataset Train Test Split",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    f = plt.figure(figsize=(45,38))
    axe = f.add_subplot(111)

    for i, df in enumerate(dfall) : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=1,
                      edgecolor='black',
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots
        


    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                 
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, fontsize=60, rotation=0)
   
    axe.set_ylabel("Number Of Comments\n", fontsize=60)
    axe.tick_params(axis='y', labelsize=60)

    # Add invisible data to add another legend
    # my_colors = ["b", "b", "b", "g", "g", "g"]
    # for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
    #     ticklabel.set_color(tickcolor)
    
    for axis in ['top','bottom','left','right']:
        axe.spines[axis].set_linewidth(4)

    l1 = axe.legend(h[:n_col], l[:n_col],  prop={'size': 60})
  
    axe.add_artist(l1)
    return axe



def load_comments(csv_path):
    df = pd.read_csv(csv_path) #pd.read_csv(csv_path) 
    df = df.drop_duplicates(subset=["Comments"], keep='last', inplace=False)
    #df.drop('Chenlu', axis=1, inplace=True)
    #df.drop('Khiem', axis=1, inplace=True)
   
    if 'twitter' in csv_path: 
        df.drop('Noushin', axis=1, inplace=True)
    df = df.dropna()

   
    #df.drop(df.loc[df['Topic']=="ukraine war"].index, inplace=True)
    #df.drop(df.loc[df['Topic']=="Biden Afghanistan"].index, inplace=True)
    #df.drop(df.loc[df['Topic']=="George Floyd"].index, inplace=True)

    # df.drop(df.loc[df['Topic']=="Ukraine War"].index, inplace=True)
    # df.drop(df.loc[df['Topic']=="Biden Afghanistan"].index, inplace=True)
    # df.drop(df.loc[df['Topic']=="George Floyd"].index, inplace=True)   
    # df.drop(df.loc[df['Topic']=="Charlottesville Rally"].index, inplace=True)
    # df.drop(df.loc[df['Topic']=="US Soldier Russian Bounty"].index, inplace=True)
    # df.drop(df.loc[df['Topic']=="TikTok Ban"].index, inplace=True)
    # df.drop(df.loc[df['Topic']=="Trump First Impeachment"].index, inplace=True)
    # df.drop(df.loc[df['Topic']=="Trump Indictment"].index, inplace=True)

   
    #Topic-Wise Training Here

    # df.drop(df.loc[df['Topic']=="Biden Afghanistain"].index, inplace=True)
    # df.drop(df.loc[df['Topic']=="charlottesville_rally"].index, inplace=True)
    # df.drop(df.loc[df['Topic']=="us_soldier_russian_bounty"].index, inplace=True)
    # df.drop(df.loc[df['Topic']=="ukraine war news"].index, inplace=True)
    # df.drop(df.loc[df['Topic']=="Trump First Impeachment"].index, inplace=True)
    #df.drop(df.loc[df['Topic']=="Murder of George Floyd"].index, inplace=True)
      
    all_comments = df[['Comments']].values.squeeze()
    all_labels = df[['Label']].values.squeeze().astype(int)
    all_topic = df[['Topic']].values.squeeze()
    all_title = df[['Title']].values.squeeze()
    all_id = df[["ID"]].values.squeeze()

    print(np.unique(all_labels, return_counts=True))

    # unique_topics = np.unique(all_title)
    # one_title_idx = np.where(all_title == unique_topics[3])

    # all_comments = df[['Comments']].values.squeeze()[one_title_idx]
    # all_labels = df[['Label']].values.squeeze()[one_title_idx]
    # all_topic = df[['Topic']].values.squeeze()[one_title_idx]
    # all_title = df[['Title']].values.squeeze()[one_title_idx]
    # all_id = df[["ID"]].values.squeeze()[one_title_idx]


    if "Transcript" in df.columns:    
        all_transcripts = df[["Transcript"]].values.squeeze()
    else:
        all_transcripts = []

    
    
    '''
    plt.figure()    
    plt.xticks(rotation=10)
    plt.xlabel("Topic")
    plt.ylabel("No. of comments")
    plt.title("Dataset Comments Per Topic")
    plt.bar(topics, comment_count_by_topic)
    plt.savefig("vis/topic_dataset_summary.jpg")
    plt.close("all")
    

    labels, comment_count_by_labels = np.unique(all_labels, return_counts=True)
    labels = ["Not Whataboutism", "Whataboutism"]

    plt.figure()    
    plt.xticks(rotation=10)
    plt.xlabel("Label")
    plt.ylabel("No. of comments")
    plt.title("Dataset Comments Per Label")
    plt.bar(labels, comment_count_by_labels)
    plt.savefig("vis/label_dataset_summary.jpg")
    plt.close("all")
    '''
   


    
    unique_id = np.unique(all_id)

    df = df.reset_index().set_index("Topic")  
    topics, comment_count_by_topic = np.unique(df.index, return_counts=True)
    
    
    
    
    
    print(topics)
    print(comment_count_by_topic)
    
    train_num = [ int(x * 0.6) for x in comment_count_by_topic]
    test_num = [ int(x * 0.2) for x in comment_count_by_topic]
    val_num = [ int(x * 0.2) for x in comment_count_by_topic]

    percent_count = []
    for i in topics: 
        labels = df.loc[i]["Label"].values
        _ , counts_by_labels = np.unique(labels, return_counts=True)
        percentage = counts_by_labels[1] / np.sum(counts_by_labels)
        percent_count.append(percentage)
   
    train_num_pos = np.array([  round(train_num[idx] * (percent_count[idx])) for idx in range(len(train_num))])
    train_num_neg = np.array([  round(train_num[idx] * (1-percent_count[idx])) for idx in range(len(train_num))])

    test_num_pos = np.array([  round(test_num[idx] * (percent_count[idx])) for idx in range(len(test_num))])
    test_num_neg = np.array([  round(test_num[idx] * (1-percent_count[idx])) for idx in range(len(test_num))])

    val_num_pos = np.array([  round(val_num[idx] * (percent_count[idx])) for idx in range(len(val_num))])
    val_num_neg = np.array([  round(val_num[idx] * (1-percent_count[idx])) for idx in range(len(val_num))])

    # # # create fake dataframes
    # topics =["Biden\nAfghanistan", "George\nFloyd", "Trump\nFirst\nImpeachment", "Charlottesville\nRally", "Ukraine\nWar\nNews", "US Soldier\nRussian\nBounty"] #[ x.replace(' ', '\n') for x in topics]
    # df1 = pd.DataFrame( np.hstack( (train_num_neg.reshape(-1,1), train_num_pos.reshape(-1,1))) , 
    #                 index=topics,
    #                 columns=["NW", "W"])
    

    # df2 = pd.DataFrame( np.hstack( (test_num_neg.reshape(-1,1), test_num_pos.reshape(-1,1))) , 
    #                 index=topics,
    #                 columns=["NW", "W"])
    # df3 = pd.DataFrame( np.hstack( (val_num_neg.reshape(-1,1), val_num_pos.reshape(-1,1))) , 
    #                 index=topics,
    #                 columns=["NW", "W"])

    # # Then, just call :
    # ax = plot_clustered_stacked([df1, df2, df3])
    # ax.figure.savefig('dataset_split.png')
  

    #Let's modify the transcripts
    
    
    # if len(all_transcripts) > 0:      
        
    #     #titles = ["The Five' slams House Democrats' 'impeachment 2.0'", "Joe Biden Responds To Report Of Russian Bounties On U.S. Troops | NBC News" ]
    #     titles = ["Trump First Impeachment", "us_soldier_russian_bounty"]
        
    #     transcripts = [["emma kratz aren't letting go of impeachment even while the pandemic is going on.", "they're still trying to get their hands on redacted grandy jury material from the muller probe.", "Here's donald trump sounding off earlier.", "nobody's been abused by pfizer like the president of the united states.", "the top of the fbi, they were dirty cops, they were crooked bad people.", "These people, they broke the law.", "What they've done to general flynn should never happen what they've done to the presidency and what they've done to this country can never be allowed to happen again." "Our administration had a laser focus on mike flynn and former fbi director james comey even suggested sensitive information on russia not be shared with flynn." "all right, judge janine this this never stops what do you think's going on now", "well you know what i think is fascinating jessie is that the the the email that was written by susan rice on the day that the president trump was being sworn in he's nothing more than a self-serving statement which is generally not allowed in a courtroom if you're the defendant but it also to me", "as a you know as a former prosecutor is an admission it is consciousness of guilt it is building a defense for barack obama when in fact no one has asked it's like the police come to your house and you say i didn't shoot my wife", "okay but she's upstairs and she's dead of a gunshot wound why are you even phrasing it like that and the other part of this is that you know when you read what they say she says the president is not initiating or instructing anything.", "and that the law enforcement needs to do what it does by the book and be mindful that we cannot entertain or share information possibly about russia with the trump administration that's basically saying that obama said", "consider not giving this information to trump i mean she couldn't she could only say it clearly more clearly if she left one word and by by the democrats trying to get the grand jury testimony of the muller investigation", "first of all think of the fact that this was the most anti-trump organization invested investigation going and secondly there's nothing else in there they found there was no collusion", "so it's all just building up to getting trump out of office dana what do you think well i think that impeachment 2.0 is actually going to be impeachment infinity because as long as president trump is in office the democrats will say that they are trying to impeach him", "or looking into impeachment and even i mean basically having their rear ends handed to them in january on the vote", "they still continue to pursue it and partly that's because their base the left wing will demand that and so they're trying to you know fulfill that demand on the grand jury testimony", "i had trey gowdy on the daily briefing today and he made a great point i'm sure the judge would agree is that grand jury testimony is secret for a reason because prosecutors take the grand jury testimony and they indict or they don't", "there's no middle ground like you don't get to go in and look at grand jury testimony just because you're curious and because you want to do something political", "now i do think that lindsey graham's hearing that he's planning to hold on the senate side will probably get a lot more answers than anything else and these you know were declassifying emails", "we have lists now there's a lot more information now than they even had in january when they were trying to impeach him in the senate", "yeah that graham investigation can't come soon enough gregg impeachment infinity", "that sounds like a nice line yeah it reminds me of like how i got i just lost track of all the star wars sequels you know after the second one i just stopped", "but the real that i guess the truth is the democrats would rather beat trump than a virus ", "they think about this you have an election in six months god knows what's gonna happen before them", "so why impeachment i think it's because they actually fear good news they know that possibly the bet the next phase will be incredible for america", "as we get back to work there's gonna be an unbridled sense of optimism in hope and that's gonna help trump because he's the king of optimism and they can't have that so the they need to combat good news with this impeachment which is nothing but a smear and it exposes their naked priorities", "what kind of naked priorities do you have mr.williams ", "oh i believe in the constitution and the founding fathers and that congress is an  equal branch of government with every right to request what they need in terms of oversight and that if there's a conflict we have the judiciary to settle the issue and in this case what we know is that the us court of appeals ruled in march" , "jessie that congress has a right to see the full report and that what we know now is that in fact it's trump's justice department that has redacted the mueller report won't let people see it", "apparently out of fear that it will reveal obstruction in terms of the roger stone case","in case now michael and that's why they're firing lightly confident is to report it no no it's the congress of the united states that has the legal right to this information", "but i guess people want to cover up for the president no matter what no that's that's absolutely not the case we're all here all right oh yeah no no see you know you've actually seen too much " ], ["and i'd like to start today by briefly addressing well i consider to be a horrifying revelation in the new york times", "last night assuming the times report is accurate they report the us intelligence community has has assessed that a russian military intelligence unit the same unit that was behind the assassination of the former kgb agent in london five years ago has been offering bounties to extremist groups in afghanistan to kill us troops", "there is no bottom to the depth of vladimir putin and their kremlin's depravity ", "if that's true it's truly shocking revelation", "if the timed report is true i emphasize again is that president trump the commander-in-chief of american troops serving in a dangerous theater of war has known about this for months ", "according to the times and done worse than nothing not only has he failed to sanction or impose any kind of consequences on russia for this egregious violation of international law", "donald trump has continued his embarrassing campaign of deference and debasing himself before vladimir putin", "he had has this information according to the times and yet he offered the host putin in the united states and sought to invite russia to rejoin the g7", "he's in his entire presence has been a gift to putin but if this is beyond the pale it's a betrayal of the most sacred duty we bear", "as a nation to protect and equip our troops when we send them in the harm's way", "that's betrayed it's a betrayal of every single american family with a loved one serving in afghanistan or anywhere   overseas", "and i'm quite frankly outraged by the report and if i'm elected president make no mistake about it vladimir putin will be confronted and i will impose serious cost on russia", "but i don't just think about this as a candidate for president i think about this as a dad a father who sent his son to serve in harm's way for a year in the middle east and in iraq and i'm disgusted on behalf of those families whose loved ones are serving today", "when your child volunteers to serve they're putting their life on the line for the country they take risk known and unknown for this nation but they should never never never ever face a threat  like this with their commander-in-chief", "turning a blind eye to a foreign power putting a bounty on their heads if i'm president this is so many other abuses will not stand", "hey nbc news viewers thanks for checking out our youtube channel subscribe by clicking on that button down here and click on any of the videos over here to watch the latest interviews show highlights and digital exclusives thanks for watching."]]
    #     all_transcripts = [item for sublist in transcripts for item in sublist]
    #     title_to_transcripts = dict(zip(titles, transcripts))
        
    #     sent_to_related = {}
        
       
    #     for idx, key in enumerate(title_to_transcripts):           
    #         transcript = title_to_transcripts[key]             
    #         relevant = list( df[ df["Topic"] == key ]["Comments"].values )
    #         for sent in transcript:          
    #             sent_to_related[sent] = relevant
    # else: 
    #     sent_to_related = {}
    
    

    return all_comments, all_labels, all_topic, all_title, all_id, unique_id, {}, all_transcripts, df

def get_items_from_split(idx, all_comments, all_labels, all_topics, all_titles, all_ids, all_transcripts, feats):
    
    return feats[idx], all_comments[idx], all_labels[idx], all_topics[idx], all_titles[idx], all_ids[idx], all_transcripts[idx]


def train_split_balance(all_comments, all_topics, all_labels, percentage=0.8):
    test_df = pd.DataFrame( list(zip(all_comments, all_topics, all_labels)), columns=["Comment", "Topic", "Labels"] )

    vid_topics = test_df["Topic"].unique()
    all_neg = 0
    all_pos = 0

    train_idx_all = []
    test_idx_all = []

    for topic in vid_topics:
        
        topic_index = test_df[ test_df["Topic"] == topic ].index 
        topic_labels = test_df[ test_df["Topic"] == topic ]["Labels"].values

        # Let's try to split this deterministically, no randomness
        neg_num, pos_num = np.bincount(topic_labels)
        positive_examples = np.array(topic_index[np.where(topic_labels==1)])
        negative_examples = np.array(topic_index[np.where(topic_labels==0)])
        all_pos += len(positive_examples)
        all_neg += len(negative_examples)
        
        train_idx = np.hstack((positive_examples[0:int(pos_num*percentage)], negative_examples[0:int(neg_num*percentage)]))
        val_idx =  np.hstack((positive_examples[int(pos_num*percentage):], negative_examples[int(neg_num*percentage):]))
    

        train_idx_all.append(train_idx)
        test_idx_all.append(val_idx)

    train_idx_all = np.hstack(train_idx_all)
    test_idx_all = np.hstack(test_idx_all)
    
    train_idx_all = np.unique(train_idx_all)
    test_idx_all = np.unique(test_idx_all)

    # Plot the dataset-split train vs. test
    
    '''
    labels = ["Not Whataboutism", "Whataboutism"]
    _, train_comment_count_by_labels = np.unique(all_labels[train_idx_all], return_counts=True)
    plt.figure()    
    plt.xticks(rotation=10)
    plt.xlabel("Label")
    plt.ylabel("No. of comments")
    plt.title("Train Set Comments Per Label")
    plt.bar(labels, train_comment_count_by_labels)
    plt.savefig("vis/train_label_dataset_summary.jpg")
    plt.close("all")

    _, test_comment_count_by_labels = np.unique(all_labels[test_idx_all], return_counts=True)
    plt.figure()    
    plt.xticks(rotation=10)
    plt.xlabel("Label")
    plt.ylabel("No. of comments")
    plt.title("Test Set Comments Per Label")
    plt.bar(labels, test_comment_count_by_labels)
    plt.savefig("vis/test_label_dataset_summary.jpg")
    plt.close("all")
    '''
    print("Non-Whatabout Samples: {}, Whatabout Samples: {}".format(all_neg, all_pos))
   
    return train_idx_all, test_idx_all 

def sort_lda_words(tup): 
  
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using second element of 
    # sublist lambda has been used 
    tup.sort(key = lambda x: x[1], reverse=True) 
    return tup


def get_unique_words(lst):
    d = {}
    for tpl in lst:
        first,  last = tpl
        if first not in d or last > d[first][-1]:
            d[first] = tpl
    
    return [*d.values()]



def get_token_split(train_idx, test_idx,  tokens):

    return [tokens[i] for i in train_idx], [tokens[i] for i in test_idx]


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub('RT','', parsed_text) #Some RTs have !!!!! in front of them
    parsed_text = re.sub(emoji_regex,'',parsed_text) #remove emojis from the text
    parsed_text = re.sub('…','',parsed_text) #Remove the special ending character is truncated
    parsed_text = re.sub('#[\w\-]+', '',parsed_text)
    # parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text


def preprocess_clean(text_string, remove_hashtags=True, remove_special_chars=True):
    # Clean a string down to just text
    # text_string=preprocess(text_string)

    parsed_text = preprocess(text_string)
    # parsed_text = parsed_text.lower()
    parsed_text = re.sub('\'', '', parsed_text)
    parsed_text = re.sub('|', '', parsed_text)
    parsed_text = re.sub(':', '', parsed_text)
    parsed_text = re.sub(',', '', parsed_text)
    parsed_text = re.sub('/', ' ', parsed_text)
    parsed_text = re.sub("\*", '', parsed_text)
    parsed_text = re.sub(';', '', parsed_text)
    parsed_text = re.sub('\.', '', parsed_text)
    parsed_text = re.sub('&amp', '', parsed_text)
    parsed_text = re.sub('ð', '', parsed_text)

    if remove_hashtags:
        parsed_text = re.sub('#[\w\-]+', '', parsed_text)
    if remove_special_chars:
        # parsed_text = re.sub('(\!|\?)+','.',parsed_text) #find one or more of special char in a row, replace with one '.'
        parsed_text = re.sub('(\!|\?)+','',parsed_text)
    return parsed_text

def add_augmentation(comments, labels, topics, titles, ids, aug_path="../dataset/augment.csv", dataframe=None):
    
    wabt_comments = np.where(labels==1)[0]
    aug_df = pd.read_csv(aug_path, index_col="Comments")

    # Dataframe to look for index
    aug_to_idx = {}
    
    for i in range(len(comments)): 
        comment = comments[i]
        label = labels[i]
        try: 
            aug_comments = aug_df.loc[comment][6:].values
          
            comment_index = np.where(dataframe["Comments"].values == comment)[0]
            sim_idx = dataframe.iloc[comment_index]["Index"].values
            for j in aug_comments:
                aug_to_idx[j] = sim_idx
            

            topic = aug_df.loc[comment]["Topic"]
           
            title = aug_df.loc[comment]["Title"]
            id = aug_df.loc[comment]["ID"]

            extend_labels = np.repeat(label, aug_comments.shape)
            extend_topic = np.repeat(topic, aug_comments.shape)
            extend_title = np.repeat(title, aug_comments.shape)
            extend_id = np.repeat(id, aug_comments.shape)

            comments = np.hstack((comments, aug_comments))
            labels = np.hstack((labels, extend_labels))
            topics = np.hstack((topics, extend_topic))
            titles = np.hstack((titles, extend_title))
            ids = np.hstack((ids, extend_id))
         
        except Exception as e :
            continue
    
    return comments, labels, topics, titles, ids, aug_to_idx

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def train_test_split_helper(comments, titles, labels, topics, ids, percentage=0.8):

    train_idx, test_idx  = train_split_balance(comments, topics, labels, percentage=percentage)
    
    train_comments = comments[train_idx]
    test_comments = comments[test_idx]
    
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    train_topics = topics[train_idx]
    test_topics = topics[test_idx]

    train_titles = titles[train_idx]
    test_titles = titles[test_idx]

    train_ids = ids[train_idx]
    test_ids = ids[test_idx]

    return train_comments, train_labels, train_topics, train_titles, train_ids, test_comments, test_labels, test_topics, test_titles, test_ids, np.array(train_idx), np.array(test_idx)