import git
from google_drive_downloader import GoogleDriveDownloader as gdd
import pandas as pd
import numpy as np
from functools import partial
from pathos.threading import ThreadPool as Pool
import numpy as np
import urllib
import os
import os.path                     
import time
import math
import cv2

class Crawler:
    def __init__(self, urls, target, threads=50):
        self.target = target
        if not os.path.exists(target):
            os.makedirs(target)
                
        self.threads = threads
        self.pool = Pool(nodes=self.threads)
        self.urls = urls    
        self.size = len(urls)
            
        
    def download(self):
        params = []
        self.start_time = time.time()
        
        for row in self.urls:
            url = row.get('url')
            filename = row.get('filename')
            filename = "{}/{}".format(self.target, filename)
            params.append((url, filename))
            
           
        self.pool.map(self.__wrapper, params)
        print("Download of {} files performed in {}s".format(len(self.urls),(time.time()-self.start_time)))
    
    
    def __wrapper(self, args):
        return self.__worker(*args)

    def __worker(self, url, filename):
        if not os.path.isfile(filename):
            try:
                print("{}:{}".format(url, filename))
                urllib.urlretrieve(url, filename)
            except Exception as e:
                print("warning: could not download {}: {}".format(url, e))
        


class AADB:
      
    def __init__(self, target):
        self.target = "{}/AADB".format(target)
            
    def download_metadata(self): 
        self.download_all()
            
    def download_all(self):
        gdd.download_file_from_google_drive(file_id='1Viswtzb77vqqaaICAQz9iuZ8OEYCu6-_',
                                    dest_path="{}/train/{}".format(self.target, 'aadb_images_train.zip'),
                                    unzip=True)
        gdd.download_file_from_google_drive(file_id='115qnIQ-9pl5Vt06RyFue3b6DabakATmJ',
                                    dest_path="{}/test/{}".format(self.target, 'aadb_images_train.zip'),
                                    unzip=True)
        gdd.download_file_from_google_drive(file_id='0BxeylfSgpk1MZ0hWWkoxb2hMU3c',
                                    dest_path="{}/labels/{}".format(self.target, 'aadb_labels.zip'),
                                    unzip=True)
        
        print("AADB dataset metadata downloaded.")
        
        

class AVA:
    
    def __init__(self, target):
        self.target = target
        
            
    def download_metadata(self):
        try:
            git.Git(self.target).clone("https://github.com/jenslaufer/ava_downloader.git")
        except:
            pass
        print("AVA dataset metadata downloaded.")
         
        
    def download_images(self, total_num_files=None):  
        if len(os.listdir("{}/ava_downloader/AVA_dataset/images".format(self.target))) > 0:
            print("AVA dataset images downloaded.")
        else:
            print("AVA dataset images need to be downloaded manually.")
        
            
            
class AROD:
    
        
    def __init__(self, target):
        self.target = target
        
    
            
    def download_metadata(self):
        try:
            git.Git(self.target).clone("https://github.com/cgtuebingen/will-people-like-your-image.git")
        except:
            pass
        
        print("AROD dataset metadata downloaded.")
    
    def download_images(self):
        print("loading metadata...")
        df = pd.read_csv("{}/will-people-like-your-image/arod/list.txt".format(self.target),sep = ";", 
                         header=None, names=['url','favs','views'])[['url']]
        df['filename'] = df.apply(lambda x: x.url[x.url.rfind("/")+1:], axis=1)
        
        print("loading metadata finished.")
        print("downloading images...")
        Crawler(df.to_dict('records'), "{}/will-people-like-your-image/arod/img/".format(self.target)).download()