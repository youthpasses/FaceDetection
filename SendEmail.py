# !/usr/local/bin/python
# -*- coding:utf-8 -*-
import httplib
import urllib

host = "106.ihuyi.com"
sms_send_uri = "/webservice/sms.php?method=Submit"

# 用户名是登录ihuyi.com账号名（例如：cf_demo123）
account = "C06588355"
# 密码 查看密码请登录用户中心->验证码、通知短信->帐户及签名设置->APIKEY
password = "600b1de0ad8abb0d7ed2cb6c0bdec999"


def send_sms(text, mobile):
    params = urllib.urlencode(
        {'account': account, 'password': password, 'content': text, 'mobile': mobile, 'format': 'json'})
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    conn = httplib.HTTPConnection(host, port=80, timeout=30)
    conn.request("POST", sms_send_uri, params, headers)
    response = conn.getresponse()
    response_str = response.read()
    conn.close()
    return response_str


if __name__ == '__main__':
    mobile = "15201625089"
    text = "您的验证码是：121254。有人在用你的电脑。"

    print(send_sms(text, mobile))