# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:28:45 2023

@author: HP
"""
import requests
import parsel
import csv
import json

f = open('nanshanqu.csv', mode = 'a', encoding = 'utf-8', newline = '')
csv_writer = csv.DictWriter(f, fieldnames = [
        '标题',
        '小区名字', 
        '地段', 
        '总价', 
        '单价', 
        '经纬度',
        '户型', 
        '面积', 
        '朝向', 
        '装修', 
        '楼层', 
        '时间', 
        '架构', 
        '详情'
    ])
csv_writer.writeheader()

for page in range(1, 101):
    print(f'正在采集第{page}页的数据')
    url = f'https://sz.lianjia.com/ershoufang/nanshanqu/pg{page}/'
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}
    
    response = requests.get(url = url, headers = headers) 
    
    selector = parsel.Selector(response.text)
    lis = selector.css('.sellListContent li .info')
    for li in lis:
        title = li.css('.title a::text').get()    
        area_info = li.css('.positionInfo a::text').getall()
        area_1 = area_info[0]
        area_2 = area_info[1]
        Price = li.css('.totalPrice span::text').get()
        unitPrice = li.css('.unitPrice span::text').get()
        houseInfo = li.css('.houseInfo::text').get()
        if len(houseInfo.split(' | ')) == 7:
            date = houseInfo.split(' | ')[5]
        else:
            date = 'None'
        house_type = houseInfo.split(' | ')[0]
        house_area = houseInfo.split(' | ')[1]
        face = houseInfo.split(' | ')[2]
        renovation = houseInfo.split(' | ')[3]
        fool = houseInfo.split(' | ')[4]
        framework = houseInfo.split(' | ')[-1]
        link = li.css('.title a::attr(href)').get()
        
        mc = area_1
        location1 = '深圳' + mc
        url2 = 'https://restapi.amap.com/v3/geocode/geo'
        para = {
            'key':'7565a4d29c0afc21662680fcf74fbb45' ,
            'address': location1
            }
        response2 = requests.get(url2, para)
        result = json.loads(response2.text)
        ll = result['geocodes'][0]['location']
        
        dit = {
            '标题': title,
            '小区名字': area_1, 
            '地段': area_2, 
            '总价': Price, 
            '单价': unitPrice, 
            '经纬度': ll,
            '户型': house_type, 
            '面积': house_area, 
            '朝向': face, 
            '装修': renovation, 
            '楼层': fool, 
            '时间': date, 
            '架构': framework, 
            '详情': link
            }
        csv_writer.writerow(dit)

f.close()