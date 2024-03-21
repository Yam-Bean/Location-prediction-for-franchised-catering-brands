# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:07:51 2023

@author: HP
"""

import requests
import pprint
import csv
import time

f = open('南山区饮品.csv', mode = 'a', encoding = 'utf-8', newline = '')
csv_writer = csv.DictWriter(f, fieldnames=[
    '店铺名',
    '评分', 
    '商圈',
    '人均消费', 
    '店铺类型', 
    '评论量', 
    '纬度', 
    '经度', 
    '详情页',
    ])
csv_writer.writeheader()

url = 'https://apimobile.meituan.com/group/v4/poi/pcsearch/30'
for page in range(0, 641, 32 ):
    data = {
            'uuid': '32289e60ccbd48699646.1680417772.1.0.0',
            'userid': '1926628026',
            'limit': '32',
            'offset': page,
            'cateId': '21329',
            'q': '快餐',
            'token': 'AgFoIb3ouYUP4kZHJUxTGkw76YSm3j4DgZSRSqp8tOC17_jmGoKxZjNbpMQkz-m8MxI7WvcZC8Sf_gAAAABsFwAAwXB-r5wTqfu-fTsKFNW5TBMQbhhj8cZ_NAvLyBFqQffArPIudrwZtPOlE4RPbAGN', 
            'areaId': '30'
            }
    headers = {
            'Referer': 'https://sz.meituan.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
        }
    response = requests.get(url = url, params = data, headers = headers)
    #print(response.json())
    # pprint.pprint(response.json())
    
    searchResult = response.json()['data']['searchResult']
    for index in searchResult:
        href = f'https://www.meituan.com/meishi/{index["id"]}/'
        dit = {
            '店铺名': index['title'], 
            '评分': index['avgscore'], 
            '商圈': index['areaname'],
            '人均消费': index['avgprice'], 
            '店铺类型': index['backCateName'], 
            '评论量': index['comments'], 
            '纬度': index['latitude'], 
            '经度': index['longitude'],
            '详情页': href
        }
        csv_writer.writerow(dit)

f.close()


