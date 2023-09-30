import argparse
import json
import pandas as pd
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi

def get_video_id(video_res):
    return video_res['snippet']['resourceId']['videoId']

def search_videos(query, collector, results=10):
    req = collector.search().list(q=query, part='snippet', type='video', order='viewCount',
                                maxResults=results)  # Max 5 results, highest is 50 results, Searching for videos
    res = req.execute()  # JSON Format
    return res['items']

def construct_url(id):
    base_url = "https://www.youtube.com/watch?v="
    return base_url + id

def get_comment_replies(parent_id, collector):
    replies = []
    next_page_token = None
    pages = 1
    i = 0

    while i < pages:
        replies_results = collector.comments().list(parentId=parent_id, part="snippet", maxResults=100
                                                            ).execute()
        #Sort by most likes
        
        if replies_results is not None:

            replies.extend(
                [item['snippet']['textOriginal'] for item in replies_results['items']])
        else:
            break

        i += 1

    return replies	

def get_transcripts(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except:
        return {}

    return transcript

def write_to_json_batch_with_replies(zip_data, file_name):
    data = {}
    data['videos'] = []
    for video_data in zip_data:
        data['videos'].append({
                'title': video_data[0],
                'URL' : video_data[1],
                'transcript': video_data[2],
                'comments': video_data[3],
                'comments_with_replies': video_data[4],
            })

    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)

    return data

def get_comments_with_replies(video_id, collector, pages=10, results=5):

  next_page_token = None
  comments = []
  comments_with_replies = {}
  

  for _ in range(pages):
    try:
      comment_results = collector.commentThreads().list(videoId=video_id, part="snippet", maxResults=results, order="relevance",
                                                                      pageToken=next_page_token).execute()
      comments.extend(
                          [  (item['snippet']['topLevelComment']['snippet']['textOriginal'], item['snippet']['topLevelComment']['snippet']['likeCount'] ) for item in comment_results['items']])
      parent_comments =	[item['snippet']['topLevelComment']['snippet']['textOriginal'] for item in comment_results['items']]

      parent_ids = [item['snippet']['topLevelComment']['id'] for item in comment_results['items']] 
      child_comments = [ get_comment_replies(parent_id, collector) for parent_id in parent_ids ]
      parent_child_dict = dict(zip(parent_comments, child_comments))  

      for k in parent_child_dict.keys():
        comments_with_replies[k] = parent_child_dict[k]
      
      next_page_token = comment_results.get('nextPageToken')
    except HttpError as err:
      return [], []

    
  return comments, comments_with_replies

def main(args):
    
    api_key =  args.api_key if args.api_key else ""
    query_name = args.topic if args.topic else "Biden Afghanistain"
    youtube_collector = build("youtube", "v3", developerKey=api_key)
    filename = args.save_as if args.save_as else "{}.json".format(query_name)
    annotation_filename = args.annotation_as if args.annotation_as else "{}.csv".format(query_name)

    print("Searching Videos ... ")
    video_search_results = search_videos(query_name, youtube_collector)    
    
    video_titles = [result['snippet']['title'] for result in video_search_results]
    video_ids = [result['id']['videoId'] for result in video_search_results]
    
    print("Constructing Video URLS ... ")
    video_urls =  [construct_url(result) for result in video_ids]

    print("Searching Transcripts ... ")
    transcripts = [get_transcripts(video_id) for video_id in video_ids]

    print("Collecting Comments ... ")
    all_comments = []
    all_comments_with_replies = []
    for id in tqdm(video_ids):
        comments, comments_with_replies = get_comments_with_replies(id, youtube_collector, pages=10, results=10)
        all_comments.append(comments)
        all_comments_with_replies.append(comments_with_replies)

    print("Saving Data to JSON ... ")
    data_zip = list(zip(video_titles, video_urls, transcripts, all_comments, all_comments_with_replies))
    json_data = write_to_json_batch_with_replies(data_zip, filename)    

    print("Saving Data to Annotation Format ... ")   
    titles_csv = []
    urls_csv = []
    topics_csv = []
    comments_csv = []
    for video in json_data['videos'][0:7]:
       
        if len(video['comments']) > 0:
            comments_csv += [x[0] for x in video['comments']]                
            title = [video['title'] for x in range(len(video['comments']))]
            url = [video['URL'] for x in range(len(video['comments']))]
            topic = [query_name for x in range(len(video["comments"]))]
            titles_csv += title
            urls_csv += url
            topics_csv += topic    
    
    dataframe = {'Topic': topics_csv, 'Title': titles_csv,
                 'URL': urls_csv, 'Comment': comments_csv}
    df = pd.DataFrame(data=dataframe)
    df = df.set_index('Topic')
    df.to_csv(annotation_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-api", "--api_key", required=False, help = "your api key")
    parser.add_argument("-t", "--topic", required=False, help = "topic")
    parser.add_argument("-save", "--save_as", required=False, help = "file path to save as")
    parser.add_argument("-annotation", "--annotation_as", required=False, help = "file path to save annotation as")
    args = parser.parse_args()
    main(args)
