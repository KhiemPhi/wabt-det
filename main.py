import argparse
import os
import warnings
import numpy as np
import optuna
import pandas as pd
import torch
import json
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data import WhataboutismDataset, WhataboutismDatasetUnlabeled
from modeling import ContextSentenceTransformer, SentenceTransformer
from utils import load_comments, train_test_split_helper
from utils.utils import train_split_balance



os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


def load_data(args, file_path="./dataset/youtube_data_1645_sim_idx.csv", aug_path="./dataset/augment.csv", unlabel_testing=False, reproduce=False):
    
    if not args.twitter: 
        file_path="./dataset/youtube_data_1645_sim_idx.csv"
    else: 
        file_path="./dataset/twitter_data_1204.csv"
    print(file_path)

    comments, labels, topics, titles, ids, _, sent_to_related, all_transcript_sents, df  = load_comments(file_path)  # load dataset w/ transcript
    queries = np.unique(df.index)
    non_wabt_count = []
    wabt_count = []
    for query in queries:        
        non_wabt, wabt = np.bincount(df.loc[query]["Label"])
        non_wabt_count.append(non_wabt)
        wabt_count.append(wabt)
    
    df_plot = pd.DataFrame({'non-wabt': non_wabt_count,

                   'wabt': wabt_count}, index=queries)
    #df_plot.to_csv("dataset_summary.csv", index=True)

    queries = np.unique(df.index)
    title_counts = []
    for query in queries:        
        title_count = len(np.unique(df.loc[query]["Title"]))
        title_counts.append(title_count)
    
    df_plot = pd.DataFrame({'no. of videos': title_counts
                   }, index=queries)
    #df_plot.to_csv("dataset_summary_by_title.csv", index=True)
    
   
    if unlabel_testing:
        extra_comments, extra_labels, extra_topics, extra_titles, extra_ids, _, _, _, new_df = load_comments("./dataset/annotations_1500.csv") # load dataset w/o transcripts
        diff_comments = np.setdiff1d(extra_comments, comments)
        idx_of_diff_comments = [np.where(extra_comments==i)[0][0] for i in diff_comments]    
        unlabeled_topics = extra_topics[idx_of_diff_comments]
        unlabeled_titles = extra_titles[idx_of_diff_comments]
        unlabeled_ids = extra_ids[idx_of_diff_comments]
        unlabeled_test_comments = extra_comments[idx_of_diff_comments]
        unlabeled_labels = extra_labels[idx_of_diff_comments]
        unlabel_test_set = WhataboutismDataset(unlabeled_test_comments, unlabeled_labels, unlabeled_topics, unlabeled_titles, unlabeled_ids, False,  new_df, False)

              
    train_comments, train_labels, train_topics, train_titles, train_ids, test_comments, test_labels, test_topics, test_titles, test_ids, train_idx_all, test_idx_all = train_test_split_helper(comments, titles, labels, topics, ids, percentage=0.8)
    aug_to_idx_train = {}
    aug_to_idx_test = {}

   
    # Divide by 2 to get test + val
    test_idx, val_idx = train_split_balance(test_comments, test_topics, test_labels, percentage=0.9)
    train_set = WhataboutismDataset(train_comments, train_labels, train_topics, train_titles, train_ids, args.context,  df, False, train_idx_all, test_comments, aug_to_idx_train, args.random, args.agnostic, args.title, name='train', data_name=os.path.basename(file_path), tuple_path=args.tuple_path, reproduce=reproduce)
    val_set =  WhataboutismDataset(test_comments[val_idx], test_labels[val_idx], test_topics[val_idx], test_titles[val_idx], test_ids[val_idx], args.context,  df, True, val_idx, test_comments,aug_to_idx_test, args.random, args.agnostic, args.title, name='val', data_name=os.path.basename(file_path), tuple_path=args.tuple_path, reproduce=reproduce)
    test_set = WhataboutismDataset(test_comments[test_idx], test_labels[test_idx], test_topics[test_idx], test_titles[test_idx], test_ids[test_idx], args.context,  df, True, test_idx, test_comments,aug_to_idx_test, args.random, args.agnostic, args.title, name='test', data_name=os.path.basename(file_path), tuple_path=args.tuple_path, reproduce=reproduce)   
    unlabel_set = WhataboutismDatasetUnlabeled(comments=all_transcript_sents, comments_to_related=sent_to_related)

    if unlabel_testing:
        return train_set, test_set, unlabel_set, unlabel_test_set
    else: 
        return train_set, test_set, unlabel_set, val_set
    
def objective(trial: optuna.trial.Trial):

    gamma = trial.suggest_float("gamma", 0.5, 3.5) #best gamma is 3.38, best beta is 0.999``      
    beta = trial.suggest_float("beta", 0.999, 0.999) #best gamma is 3.38, best beta is 0.999``     
 
    train_set, test_set, unlabel_set, val_set = load_data(args, reproduce=False)

      
    checkpoint_callback = ModelCheckpoint(
        monitor="validation-f1",
        dirpath="best_ckpts",
        filename="wabt-det-{epoch:02d}-{validation-f1:.2f}" +  "-" + args.study_name,
        save_top_k=1,
        mode="max",
    )
    if args.context:                  
        model = ContextSentenceTransformer(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0.9   , gamma=0.5,class_num=2, context=args.context, loss=args.loss, cross=False, unlabel_set=unlabel_set)
    
    
    else: 
        model = ContextSentenceTransformer(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0.9   , gamma=1.85,class_num=2, context=args.context, loss=args.loss, cross=False, unlabel_set=unlabel_set)

    trainer = Trainer(devices=[int(args.gpu)] if torch.cuda.is_available() else 0,  accelerator="gpu",
                      max_epochs=args.epochs, auto_select_gpus=True, benchmark=True,        
                      auto_lr_find=True, check_val_every_n_epoch=1, num_sanity_val_steps=0, callbacks=[checkpoint_callback], logger=True, accumulate_grad_batches=4)
   
    
    trainer.fit(model)   
    if not args.twitter: 
        file_path="./dataset/youtube_data_1645_sim_idx.csv"
        json_folder = args.tuple_path
    else: 
        file_path="./dataset/twitter_data_1204.csv"
        json_folder = args.tuple_path
    data_name = os.path.basename(file_path)
                
    for name in ["train", "test", "val"]: 
        json_file_name = data_name[:-4] + "_" + name + ".json"
        json_file_name_new = "{}_f1={:.2f}.json".format( json_file_name[:-5], (float(model.best_f1))  )
        json_file_name_new = os.path.join(json_folder, json_file_name_new)                   
        with open(json_file_name_new, "w")  as out_file:
            if name == 'train':
                json.dump(train_set.comments_to_idx, out_file)                            
            elif name == 'test':
                json.dump(test_set.comments_to_idx, out_file)
            else: 
                json.dump(val_set.comments_to_idx, out_file)                    
                    

    return trainer.callback_metrics["best-f1"].item()


def main(args):
    if not args.testing:

        source = open('{}.txt'.format(args.study_name), 'w')
      
        if args.loss == "focal" or args.loss == "softmax":   
            pruner: optuna.pruners.BasePruner = (
                optuna.pruners.NopPruner()
            )
            study = optuna.create_study(direction="maximize", pruner=pruner, study_name=args.study_name)
            study.optimize( objective, n_trials=1, gc_after_trial=True)
            
            print("Number of finished trials: {}".format(len(study.trials)), file=source)

            print("Best trial:", file=source)
            trial = study.best_trial

            print("Validation-F1: {}".format(trial.value), file=source)

            print("Params Optimized: ", file=source)
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value), file=source)
            
            f1_results = []
            for i in study.trials:
                f1_results.append(i.value)
            f1_results = np.array(f1_results)

            print(np.mean(f1_results))
            print(np.std(f1_results)) 

            # Read vis/validation_acc_tab
            with open('vis/validation_acc_tab.txt') as meta_acc_file, open('{}.txt'.format(args.study_name)) as res_file:
                for line in meta_acc_file:
                    print(line, file=source)
                   
                    
             
        else:
            for i in range(0,args.num_trains):
                train_set, test_set, unlabel_set, val_set = load_data(args, reproduce=False)
                if args.context:
                    model = ContextSentenceTransformer(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0, gamma=0, class_num=2, context=args.context, loss=args.loss, cross=False)
                else: 
                    model = SentenceTransformer(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0, gamma=0, class_num=2, context=args.context, loss=args.loss)
                    
                checkpoint_callback = ModelCheckpoint(
                    monitor="validation-f1",
                    dirpath="/data/kphi/best_ckpts",
                    filename="wabt-det-{epoch:02d}-{validation-f1:.2f}" +  "-" + args.study_name,
                    save_top_k=1,
                    mode="max",
                )
                trainer = Trainer(devices=[int(args.gpu)] if torch.cuda.is_available() else 0,  accelerator="gpu",
                                max_epochs=args.epochs, auto_select_gpus=True, benchmark=True,        
                                auto_lr_find=True, check_val_every_n_epoch=1, num_sanity_val_steps=0, callbacks=[checkpoint_callback], logger=False)
                
                
                trainer.fit(model) 
                if not args.twitter: 
                    file_path="./dataset/youtube_data_1645_sim_idx.csv"
                    json_folder = args.tuple_path
                else: 
                    file_path="./dataset/twitter_data_1204.csv"
                    json_folder = args.tuple_path
                
                results_summary = "trial_{}_validation_acc_tab_f1={}.txt".format(   i+1,  (float(model.best_f1))  )
                results_summary = os.path.join(json_folder, results_summary)
                source = open(results_summary, 'w')
                print("Validation-F1: {}".format(model.best_f1), file=source)                
                
                with open('vis/validation_acc_tab.txt') as meta_acc_file, open('{}.txt'.format(args.study_name)) as res_file:
                    for line in meta_acc_file:
                        print(line, file=source)
                data_name = os.path.basename(file_path)
                
                for name in ["train", "test", "val"]: 
                    json_file_name = data_name[:-4] + "_" + name + ".json"
                    json_file_name_new = "{}_f1={:.2f}.json".format( json_file_name[:-5], (float(model.best_f1))  )
                    json_file_name_new = os.path.join(json_folder, json_file_name_new)                   
                    with open(json_file_name_new, "w")  as out_file:
                        if name == 'train':
                            json.dump(train_set.comments_to_idx, out_file)                            
                        elif name == 'test':
                            json.dump(test_set.comments_to_idx, out_file)
                        else: 
                            json.dump(val_set.comments_to_idx, out_file)                    
                    

    else:         
        train_set, test_set, unlabel_set, val_set = load_data(args, reproduce=True)
        if args.context:
            model = ContextSentenceTransformer(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0, gamma=0, class_num=2, context=args.context, loss=args.loss, cross=False)
        else: 
            model = SentenceTransformer(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0, gamma=0, class_num=2, context=args.context, loss=args.loss)
               
        weights_path = args.weights
        trained_model = model.load_from_checkpoint(train_set=train_set, test_set=test_set, val_set=val_set, checkpoint_path=weights_path)           

        checkpoint_callback = ModelCheckpoint(
            monitor="validation-f1",
            dirpath="/data/kphi/best_ckpts",
            filename="wabt-det-{epoch:02d}-{validation-f1:.2f}" +  "-" + args.study_name,
            save_top_k=1,
            mode="max",
        )
        
        trainer = Trainer(devices=[int(args.gpu)] if torch.cuda.is_available() else 0,  accelerator="gpu",
                        max_epochs=args.epochs, auto_select_gpus=True, benchmark=True,        
                        auto_lr_find=True, check_val_every_n_epoch=1, num_sanity_val_steps=0, callbacks=[checkpoint_callback], logger=False)
        trainer.test(trained_model)
        print(trained_model.report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=str,
                        default='0', help="GPU to use")
    parser.add_argument("-b", "--batch_size", type=int,
                        default=35, help="batch size to use")
    parser.add_argument("-nt", "--num_trains", type=int,
                        default=1, help="number of trials to train")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=1e-4, help="batch size to use")
    parser.add_argument("-ep", "--epochs", type=int,
                        default=20, help="batch size to use")
    parser.add_argument("-c", "--context", action='store_true',
                        default=False, help="use context tuples")   
    parser.add_argument("-l", "--loss", type=str,  default="focal", help="type of loss to use")
    parser.add_argument("-sn", "--study_name", type=str, default="context-aug", help="name of test/train ran")
    parser.add_argument("-t", "--testing", action='store_true',
                        default=False, help="test using pre-trained models")
    parser.add_argument("-r", "--random", action='store_true',
                        default=False, help="use random tuples")
    parser.add_argument("-a", "--agnostic", action='store_true',
                        default=False, help="use topic agnostic testing and trining")
    parser.add_argument("-ti", "--title", action='store_true',
                        default=False, help="use title as context")   
    parser.add_argument("-tw", "--twitter", action='store_true',
                        default=False, help="use twitter data")
    parser.add_argument("-w", "--weights", type=str, default="/data/kphi/best_ckpts/wabt-det-epoch=09-validation-f1=54.55-context no augments poly focal loss determinstic pairing.ckpt", help="path to weight")
    parser.add_argument("-tp", "--tuple_path", type=str, default="context_json", help="path to weight")
    
    
    args = parser.parse_args()
    main(args)
