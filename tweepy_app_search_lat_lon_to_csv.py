"""
Python wrapper for Twitter API
"""

import tweepy #https://github.com/tweepy/tweepy
import csv
import sys
import numpy as np

#Twitter API credentials
consumer_key = "XXXXXXXXXXXXXXXXXXXXXXXXX"
consumer_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
access_token = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
access_token_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True) #https://www.karambelkar.info/2015/01/how-to-use-twitters-search-rest-api-most-effectively./

if(not api):
    print("Can't Authenticate")
    sys.exit(-1)

lang = "en"

#South
location = "south" # We'll store the tweets in a csv file later with the city name.
geocode = "33.17434,-91.23046,100mi" #geocode = "lat, lon, radius"

#New England
#location = "new_england"
#geocode = "43.19716,-73.21289,100mi"

maxTweets = 15000
tweetsPerQry = 100  # this is the max the API permits

# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = None

# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = -1

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))
retweeted = 0
#outtweets = ["id","created_at","text", "entities"]
outtweets = ["id","created_at","text"]
while tweetCount < maxTweets:
    try:
        if (max_id <= 0):
            if (not sinceId):
                new_tweets = api.search(geocode=geocode, count=tweetsPerQry, lang=lang)
            else:
                new_tweets = api.search(geocode=geocode, count=tweetsPerQry,
                                        since_id=sinceId, lang=lang)
        else:
            if (not sinceId):
                new_tweets = api.search(geocode=geocode, count=tweetsPerQry,
                                        max_id=str(max_id - 1), lang=lang)
            else:
                new_tweets = api.search(geocode=geocode, count=tweetsPerQry,
                                        max_id=str(max_id - 1),
                                        since_id=sinceId, lang=lang)
        if not new_tweets:
            print("No more tweets found")
            break
        
            #transform the tweepy tweets into a 2D array that will populate the csv	
        for tweet in new_tweets:
            if (not tweet.retweeted) and ('RT @' not in tweet.text):
                outtweets = np.vstack((outtweets,[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")]))
                tweetCount += 1
                print("Downloaded {0} tweets".format(tweetCount))
            else:
                retweeted = retweeted + 1
        max_id = new_tweets[-1].id
        #max_id = outtweets[-1][0]
        
    except tweepy.TweepError as e:
        # Just exit if any error
        print("some error : " + str(e))
        break
    
#write the csv	
with open('%s_tweets.csv' % location, 'w') as f:
	writer = csv.writer(f)
	#writer.writerow(["id","created_at","text"])
	writer.writerows(outtweets)
  
print ("Downloaded {0} tweets, Saved to {1}".format((tweetCount), location))
