#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:12:16 2018

@author: joelgehin
"""

# coding: utf-8
import requests
from bs4 import BeautifulSoup

website_prefix = "https://www.reuters.com/finance/stocks/financial-highlights/"

QUOTING_CODE = {}
QUOTING_CODE["AIRBUS"]="AIR.PA"
QUOTING_CODE["LVMH"]="LVMH.PA"
QUOTING_CODE["DANONE"]="DANO.PA"


QUOTATION_FRAME="Quarter Ending Dec-18" 


for Company in ("AIRBUS","DANONE","LVMH"):
    url_page = website_prefix + QUOTING_CODE[Company]
    res = requests.get(url_page)
    if res.status_code == 200:
        html_doc =  res.text
        soup = BeautifulSoup(html_doc,"html.parser")
        specific_class = "stripe"
        exe_stripe=soup.find('tr',class_=specific_class)
        amount = exe_stripe.find_all("td")[2].text
        print(Company + " on " + QUOTATION_FRAME + " had sales of " + amount)
        exe_detail=soup.find('div',class_="sectionQuote nasdaqChange")
        amount_detail = exe_detail.find_all("span")[1].text
        exe_change=soup.find('div',class_="sectionQuote priceChange")
        pc_change = exe_change.find_all("span")[3].text
        print(Company + " now has a value of :" + amount_detail + " (variating of "+ pc_change + " )")
        exe_shares=soup.find_all('div',class_="moduleBody")[13]
        sh_owned=exe_shares.find_all("td")[1].text
        print(Company + " shares owned is " , sh_owned)
        exe_yield=soup.find_all('div',class_="moduleBody")[4]
        val_yield1=exe_yield.find_all("td")[1].text
        val_yield2=exe_yield.find_all("td")[2].text
        val_yield3=exe_yield.find_all("td")[3].text
        print(Company + " yield company" , val_yield1, " Industry" , val_yield2, " sector" , val_yield3)
