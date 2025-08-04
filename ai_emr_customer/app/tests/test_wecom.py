import requests
import pytest

# @pytest.fixture()
def get_access_token():
    company_id = "ww20735d5255ab5a9f"
    secret = "AWqiXKyA1MScmlJzLQT7KvUBs9Ie0_IEyHlEwFUxkaU"
    url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={company_id}&corpsecret={secret}"
    resp = requests.get(url)
    assert resp.status_code == 200
    # TODO cache token and expires_in
    access_token = resp.json()["access_token"]
    expires_in = resp.json()["expires_in"]
    print(access_token, expires_in)
    return {"access_token": access_token, "expires_in": expires_in}

access_token = get_access_token()["access_token"]

def get_reliable_ip():
    """获取企微服务器ip段, 企微徽调某个url"""
    url = f"https://qyapi.weixin.qq.com/cgi-bin/getcallbackip?access_token={access_token}"
    resp = requests.get(url)
    assert resp.status_code == 200
    ips = resp.json()["ip_list"]
    print(f"企微调用某个 url {ips}")
    return ips


def get_api_reliable_ip():
    """
    获取企业微信接口IP段, 开发者调用企微API ip
    """
    url = f"https://qyapi.weixin.qq.com/cgi-bin/get_api_domain_ip?access_token={access_token}"
    resp = requests.get(url)
    assert resp.status_code == 200
    ips = resp.json()["ip_list"]
    print(f"开发者调用企微API ip {ips}")
    return ips

def get_external_user():

    EXTERNAL_USERID = ""
    CURSOR = ""
    url = f"https://qyapi.weixin.qq.com/cgi-bin/externalcontact/get?access_token={acces_token}&external_userid={EXTERNAL_USERID}&cursor={CURSOR}"

    

def get_userid():
    json_data = {
	"cursor": "",
	"limit": 10000
}
    url = f"https://qyapi.weixin.qq.com/cgi-bin/user/list_id?access_token={access_token}"
    resp = requests.post(url, json=json_data)
    assert resp.status_code == 200
    print(f"获取用户id： {resp.json()}")

def get_user_by_mobile():
    json_data = {
	"mobile": "13940227903", # TangXueDuo
}
    url = f"https://qyapi.weixin.qq.com/cgi-bin/user/getuserid?access_token={access_token}"
    resp = requests.post(url, json=json_data)
    assert resp.status_code == 200
    print(f"通过手机号获取用户信息如下： {resp.json()}")

def get_external_id_info(external_id):
    cursor = ""
    url =f"https://qyapi.weixin.qq.com/cgi-bin/externalcontact/get?access_token={access_token}&external_userid={external_id}&cursor={cursor}"
    resp = requests.get(url)
    assert resp.status_code == 200
    return resp.json()

def get_wechat_users_by_userid(user_id="TangXueDuo"):
    url = f"https://qyapi.weixin.qq.com/cgi-bin/externalcontact/list?access_token={access_token}&userid={user_id}"
    resp = requests.get(url)
    assert resp.status_code == 200
    external_users = resp.json()["external_userid"] # {'errcode': 0, 'errmsg': 'ok', 'external_userid': ['wmIVWRDwAAzdpePYDr8R0yWur6jLzbog']}
    print(f"用户{user_id}的外部客户如下： {external_users}")
    # 
    for external_id in external_users:
        external_info = get_external_id_info(external_id)
        if external_info["external_contact"]["type"]==1:
            print(f"用户{user_id}的外部微信客户如下： {external_info}")

def send_msg():
    """给指定用户发消息"""
    

def receive_msg():
    pass

if __name__ == "__main__":
    wecom2external_ip = get_reliable_ip()[0]
    url2wecom_ip = get_api_reliable_ip()[0]
    # get_userid()
    get_user_by_mobile()
    get_wechat_users_by_userid()
    # get_external_user()
