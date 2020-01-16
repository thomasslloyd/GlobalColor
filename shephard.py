# Pulls in images from different sources
# Thomas Lloyd

import numpy as np
import flickrapi
import urllib.request

# make private
api_key = '55d426a59efdae8b630aaa3afbac4000'
api_secret = '72f4bde28a867f41'
keyword1 = 'tokyo'

def initialize(api_key, api_secret):
    flickr = flickrapi.FlickrAPI(api_key, api_secret)
    return flickr


def pullimages(flickr):
    # photos = flickr.photos.search(user_id='60027860@N06', per_page='10')
    photos = flickr.walk(text=keyword1,
                         tag_mode='all',
                         tags=keyword1,
                         extras='url_c',
                         per_page=500,        
                         sort='relevance')
    urls = []
    for i, photo in enumerate(photos):
        url = photo.get('url_c')
        urls.append(url)
        # get 50 urls
        if i > 1000:
            break
    return urls

def fakeurls():
    urls = []
    urls.append('https://live.staticflickr.com/7858/47443394111_c9b79def1b_c.jpg')
    urls.append('https://live.staticflickr.com/4181/34268611090_aa1b6cd86f_c.jpg')
    urls.append('https://live.staticflickr.com/4226/33953994894_7213c010f4_c.jpg')
    urls.append('https://live.staticflickr.com/4902/44209156090_48c2861574_c.jpg')
    urls.append('https://live.staticflickr.com/7328/27511837520_12d32ef9bb_c.jpg')

    for n in range(0, len(urls)):
        url = urls[n] 
        if type(url) == str:
            print("url" + str(n) + ": " + url)
    
    
    return urls


def saveimages(urls):
    print('beginning url download')
    for n in range(0, len(urls)):
        url = urls[n]
        if type(url) == str:
            # urllib.request.urlretrieve(url, '/mnt/f/amsterdam/ams' + str(n) + '.jpg')
            # urllib.request.urlretrieve(url, '/mnt/f/newyork/ny' + str(n) + '.jpg') # zero indexed
            urllib.request.urlretrieve(url, '/mnt/f/tokyo/tko' + str(n) + '.jpg') # zero indexed

        # else raise Exception('url type is not a string')


# main
flickr = initialize(api_key, api_secret)
# urls = fakeurls()
urls = pullimages(flickr)
saveimages(urls)
print('number of urls stored: ' + str(len(urls)))
print(keyword1 + ' images downloaded.')